import groovy.json.JsonSlurperClassic

def runWithOptionalLock(String lockLabel, Closure body) {
    if (lockLabel?.trim()) {
        try {
            lock(label: lockLabel, variable: 'LOCKED_ASCEND_RESOURCE') {
                echo "Acquired Ascend lock: ${env.LOCKED_ASCEND_RESOURCE}"
                body()
            }
        } catch (NoSuchMethodError err) {
            echo "Lockable Resources plugin is unavailable; running without lock for label=${lockLabel}"
            body()
        }
    } else {
        echo "ASCEND_LOCK_LABEL is empty; running without Jenkins lock."
        body()
    }
}

def tryUnstash(String stashName) {
    try {
        unstash stashName
    } catch (Exception err) {
        echo "No stash named '${stashName}' is available in this workspace: ${err.getMessage()}"
    }
}

def sanitizeRuntimeName(String value) {
    return value.replaceAll(/[^A-Za-z0-9_.-]+/, '-').replaceAll(/^-+|-+$/, '')
}

def runtimeParallelism(def pipelineParams) {
    def configured = 0
    try {
        configured = pipelineParams.RUNTIME_PARALLELISM.toInteger()
    } catch (Exception ignored) {
        configured = 0
    }
    return Math.max(configured, 0)
}

pipeline {
    agent none

    options {
        timestamps()
    }

    parameters {
        choice(name: 'CI_MODE', choices: ['pr', 'static', 'smoke', 'nightly', 'release', 'benchmark'], description: 'Jenkins CI mode')
        choice(name: 'CASE_LEVEL', choices: ['auto', 'static', 'smoke', 'nightly', 'release', 'benchmark'], description: 'DeployCase level selector')
        string(name: 'DEPLOY_CASE_GLOB', defaultValue: '.ci/deploy_cases/*.yaml', description: 'DeployCase YAML glob')
        booleanParam(name: 'RUN_ALL', defaultValue: false, description: 'Run all matched DeployCases')
        booleanParam(name: 'BUILD_IMAGE', defaultValue: false, description: 'Build CI image before validation')
        string(name: 'IMAGE_TAG', defaultValue: 'jenkins-ci', description: 'CI image tag')
        booleanParam(name: 'CHECK_DOC_DIFF', defaultValue: true, description: 'Fail if generated docs are not committed')
        booleanParam(name: 'RUN_ASCEND', defaultValue: false, description: 'Launch vLLM Ascend services')
        booleanParam(name: 'RUN_BENCHMARK', defaultValue: false, description: 'Run benchmark checks when enabled by case')
        string(name: 'MODEL_ROOT', defaultValue: '', description: 'Optional local model root on Jenkins agents')
        string(name: 'CPU_LABEL', defaultValue: 'linux', description: 'Jenkins label for CPU/static stages')
        string(name: 'DOCKER_BUILDER_LABEL', defaultValue: 'linux && docker', description: 'Jenkins label for Docker build')
        string(name: 'ASCEND_LABEL', defaultValue: 'ascend', description: 'Jenkins label for Ascend runtime stages')
        string(name: 'ASCEND_LOCK_LABEL', defaultValue: '', description: 'Optional Lockable Resources label for Ascend hosts')
        string(name: 'ASCEND_DOCKER_IMAGE', defaultValue: '', description: 'Docker image used for Ascend runtime containers')
        string(name: 'ASCEND_DOCKER_DEVICE_ARGS', defaultValue: '', description: 'Extra site-specific runtime container arguments')
        string(name: 'NPU_LOCK_DIR', defaultValue: '/tmp/vllm-ascend-ci/npu', description: 'Host-local file lock directory for Ascend cards')
        string(name: 'PORT_LOCK_DIR', defaultValue: '/tmp/vllm-ascend-ci/ports', description: 'Host-local file lock directory for runtime ports')
        string(name: 'RUNTIME_PARALLELISM', defaultValue: '0', description: 'Concurrent cases inside one runtime container; 0 means auto')
        booleanParam(name: 'DRY_RUN_RUNTIME', defaultValue: true, description: 'Validate allocation and docker commands without launching real containers/models')
        string(name: 'REGISTRY', defaultValue: '', description: 'Optional Docker registry prefix')
    }

    environment {
        PYTHONUNBUFFERED = '1'
    }

    stages {
        stage('Checkout') {
            agent { label "${params.CPU_LABEL}" }
            steps {
                checkout scm
                sh 'git status --short'
            }
        }

        stage('Precheck') {
            agent { label "${params.CPU_LABEL}" }
            steps {
                sh 'bash .ci/scripts/run_precheck.sh'
            }
        }

        stage('Validate Deploy Case Schema') {
            agent { label "${params.CPU_LABEL}" }
            steps {
                sh 'python3 .ci/scripts/validate_deploy_case.py --cases "${DEPLOY_CASE_GLOB}" --output reports/validated_cases.json'
                stash name: 'validated-report', includes: 'reports/validated_cases.json', allowEmpty: true
            }
        }

        stage('Generate Deploy Docs') {
            agent { label "${params.CPU_LABEL}" }
            steps {
                sh 'python3 .ci/scripts/render_deploy_docs.py --cases "${DEPLOY_CASE_GLOB}" --level all --output-dir docs/deploy/generated'
                script {
                    if (params.CHECK_DOC_DIFF) {
                        sh '''
                            if [ -n "$(git status --short docs/deploy/generated)" ]; then
                              git status --short docs/deploy/generated
                              echo "Generated deploy docs changed. Regenerate and commit docs/deploy/generated/."
                              exit 1
                            fi
                        '''
                    }
                }
                stash name: 'generated-docs', includes: 'docs/deploy/generated/**/*.md', allowEmpty: true
            }
        }

        stage('Build CI Image') {
            when { expression { return params.BUILD_IMAGE } }
            agent { label "${params.DOCKER_BUILDER_LABEL}" }
            steps {
                script {
                    def image = "vllm-ascend-ci:${params.IMAGE_TAG}"
                    if (params.REGISTRY?.trim()) {
                        image = "${params.REGISTRY}/vllm-ascend-ci:${params.IMAGE_TAG}"
                    }
                    sh "docker build -f .ci/docker/Dockerfile.ci -t '${image}' ."
                }
            }
        }

        stage('Select Deploy Cases') {
            agent { label "${params.CPU_LABEL}" }
            steps {
                script {
                    def runAllArg = params.RUN_ALL ? '--run-all' : ''
                    sh """
                        python3 .ci/scripts/select_deploy_cases.py \
                          --cases "${params.DEPLOY_CASE_GLOB}" \
                          --level "${params.CASE_LEVEL}" \
                          --ci-mode "${params.CI_MODE}" \
                          ${runAllArg} \
                          --output reports/selected_cases.txt
                    """
                    stash name: 'selected-cases', includes: 'reports/selected_cases.txt', allowEmpty: false
                }
            }
        }

        stage('Static Validate Selected Cases') {
            agent { label "${params.CPU_LABEL}" }
            steps {
                unstash 'selected-cases'
                sh '''
                    python3 .ci/scripts/static_validate_cases.py \
                      --case-list reports/selected_cases.txt \
                      --model-root "${MODEL_ROOT}" \
                      --output reports/static_validate.json
                '''
                stash name: 'static-reports', includes: 'reports/selected_cases.txt,reports/static_validate.json', allowEmpty: true
            }
        }

        stage('Plan Runtime Container') {
            when { expression { return params.RUN_ASCEND } }
            agent { label "${params.CPU_LABEL}" }
            steps {
                unstash 'selected-cases'
                sh '''
                    python3 .ci/scripts/plan_deploy_shards.py \
                      --case-list reports/selected_cases.txt \
                      --output reports/runtime_shards.json \
                      --shard-dir reports/runtime_shards
                '''
                stash name: 'runtime-plan', includes: 'reports/runtime_shards.json,reports/runtime_shards/*.txt', allowEmpty: false
            }
        }

        stage('Run vLLM Ascend Deploy Cases') {
            when { expression { return params.RUN_ASCEND } }
            agent none
            steps {
                script {
                    node(params.ASCEND_LABEL) {
                        checkout scm
                        unstash 'selected-cases'
                        unstash 'runtime-plan'
                        def plan = new JsonSlurperClassic().parseText(readFile('reports/runtime_shards.json'))
                        if ((plan.shards ?: []).isEmpty()) {
                            error('RUN_ASCEND=true but no runtime cases were planned.')
                        }
                        def dockerImage = params.ASCEND_DOCKER_IMAGE?.trim()
                        if (!dockerImage) {
                            dockerImage = params.REGISTRY?.trim() ?
                                "${params.REGISTRY}/vllm-ascend-ci:${params.IMAGE_TAG}" :
                                "vllm-ascend-ci:${params.IMAGE_TAG}"
                        }
                        def benchmarkArg = params.RUN_BENCHMARK ? '--run-benchmark' : ''
                        def containerName = sanitizeRuntimeName("vllm-ascend-ci-${env.BUILD_TAG}")
                        def runtimeParallel = runtimeParallelism(params)
                        def totalPorts = Math.max((plan.total_port_count ?: 1) as int, 1)
                        try {
                            catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                                withEnv([
                                    "ASCEND_DOCKER_IMAGE_RESOLVED=${dockerImage}",
                                    "ASCEND_DOCKER_DEVICE_ARGS=${params.ASCEND_DOCKER_DEVICE_ARGS}",
                                    "CONTAINER_NAME=${containerName}",
                                    "VLLM_CI_CONTAINER_NAME=${containerName}",
                                    "CONTAINER_WORKSPACE=/home/ma-user/AscendCloud/jenkins",
                                    "ALLOCATION_JSON=reports/runtime_container_allocation.json",
                                    "MODEL_ROOT=${params.MODEL_ROOT}",
                                    "NPU_LOCK_DIR=${params.NPU_LOCK_DIR}",
                                    "PORT_LOCK_DIR=${params.PORT_LOCK_DIR}",
                                    "TOTAL_PORT_COUNT=${totalPorts}",
                                    "RUNTIME_PARALLELISM_RESOLVED=${runtimeParallel}",
                                    "BENCHMARK_ARG=${benchmarkArg}",
                                    "DRY_RUN_RUNTIME=${params.DRY_RUN_RUNTIME}"
                                ]) {
                                    runWithOptionalLock(params.ASCEND_LOCK_LABEL) {
                                        sh '''#!/usr/bin/env bash
                                            set -euo pipefail
                                            mkdir -p reports/nightly/case_results reports/runtime_shards logs/deploy
                                            runtime_args=()
                                            if [ -n "${BENCHMARK_ARG}" ]; then
                                              runtime_args+=("${BENCHMARK_ARG}")
                                            fi
                                            dry_run_arg=()
                                            if [ "${DRY_RUN_RUNTIME}" = "true" ]; then
                                              dry_run_arg=(--dry-run)
                                            fi

                                            python3 .ci/scripts/run_runtime_container.py \
                                              --case-list reports/selected_cases.txt \
                                              --allocation-json "${ALLOCATION_JSON}" \
                                              --docker-image "${ASCEND_DOCKER_IMAGE_RESOLVED}" \
                                              --container-name "${CONTAINER_NAME}" \
                                              --workspace "${WORKSPACE}" \
                                              --model-root "${MODEL_ROOT}" \
                                              --card-count 0 \
                                              --port-count "${TOTAL_PORT_COUNT}" \
                                              --npu-lock-dir "${NPU_LOCK_DIR}" \
                                              --port-lock-dir "${PORT_LOCK_DIR}" \
                                              --parallelism "${RUNTIME_PARALLELISM_RESOLVED}" \
                                              --extra-docker-args "${ASCEND_DOCKER_DEVICE_ARGS}" \
                                              --print-command \
                                              --continue-on-error \
                                              "${dry_run_arg[@]}" \
                                              "${runtime_args[@]}"
                                        '''
                                    }
                                }
                            }
                        } finally {
                            sh 'bash .ci/scripts/cleanup_processes.sh logs/deploy || true'
                            stash name: 'runtime-results', includes: 'reports/nightly/case_results/**/*,logs/**/*,reports/runtime_container_allocation.json,reports/runtime_docker_command.sh', allowEmpty: true
                        }
                    }
                }
            }
        }

        stage('Run Benchmark') {
            when { expression { return params.RUN_BENCHMARK } }
            agent { label "${params.CPU_LABEL}" }
            steps {
                echo 'Benchmark execution is handled by run_deploy_cases.py for selected DeployCases.'
                echo 'Use compare_benchmark.py manually or in a follow-up stage once baseline files are populated.'
            }
        }

        stage('Generate Nightly Report') {
            agent { label "${params.CPU_LABEL}" }
            steps {
                script {
                    tryUnstash('validated-report')
                    tryUnstash('generated-docs')
                    tryUnstash('selected-cases')
                    tryUnstash('static-reports')
                    tryUnstash('runtime-plan')
                    tryUnstash('runtime-results')
                }
                sh '''
                    python3 .ci/scripts/collect_env.py --output reports/nightly/environment.json
                    python3 .ci/scripts/generate_junit_report.py \
                      --input reports/nightly/case_results \
                      --output reports/nightly/junit.xml
                    python3 .ci/scripts/generate_nightly_report.py \
                      --input reports/nightly/case_results \
                      --environment reports/nightly/environment.json \
                      --output reports/nightly/index.html
                '''
                stash name: 'nightly-reports', includes: 'reports/**/*,logs/**/*,docs/deploy/generated/**/*.md', allowEmpty: true
            }
        }

        stage('Archive reports and logs') {
            agent { label "${params.CPU_LABEL}" }
            steps {
                script {
                    tryUnstash('nightly-reports')
                }
                junit allowEmptyResults: true, testResults: 'reports/nightly/junit.xml'
                archiveArtifacts artifacts: 'reports/**/*,logs/**/*,docs/deploy/generated/**/*.md', allowEmptyArchive: true
                script {
                    try {
                        publishHTML(target: [
                            allowMissing: true,
                            alwaysLinkToLastBuild: true,
                            keepAll: true,
                            reportDir: 'reports/nightly',
                            reportFiles: 'index.html',
                            reportName: 'DeployCase Nightly Report'
                        ])
                    } catch (NoSuchMethodError err) {
                        echo 'HTML Publisher plugin is unavailable; archived HTML report as artifact instead.'
                    }
                }
            }
        }
    }
}
