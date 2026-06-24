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

def runtimeNodeCount(def pipelineParams) {
    def configured = 1
    try {
        configured = pipelineParams.RUNTIME_NODE_COUNT.toInteger()
    } catch (Exception ignored) {
        configured = 1
    }
    return Math.max(configured, 1)
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
        booleanParam(name: 'RUN_UT', defaultValue: true, description: 'Run pytest unit tests on the Ascend agent')
        string(name: 'UT_TEST_PATH', defaultValue: 'tests/ut', description: 'pytest unit test path')
        string(name: 'UT_MARK_EXPR', defaultValue: '', description: 'Optional pytest marker expression for unit tests; empty means run all UT')
        string(name: 'UT_PYTEST_ARGS', defaultValue: '-q', description: 'Additional simple pytest args for unit tests')
        string(name: 'MODEL_ROOT', defaultValue: '', description: 'Optional local model root on Jenkins agents')
        string(name: 'CPU_LABEL', defaultValue: 'linux', description: 'Jenkins label for CPU/static stages')
        string(name: 'DOCKER_BUILDER_LABEL', defaultValue: 'linux && docker', description: 'Jenkins label for Docker build')
        string(name: 'ASCEND_LABEL', defaultValue: 'ascend', description: 'Jenkins label for Ascend runtime stages')
        string(name: 'ASCEND_LOCK_LABEL', defaultValue: '', description: 'Optional Lockable Resources label for Ascend hosts')
        string(name: 'RUNTIME_NODE_COUNT', defaultValue: '1', description: 'Number of Ascend lockable resources to acquire for Ansible runtime')
        string(name: 'ASCEND_DOCKER_IMAGE', defaultValue: '', description: 'Docker image used for Ascend runtime containers')
        string(name: 'ASCEND_DOCKER_DEVICE_ARGS', defaultValue: '', description: 'Extra site-specific runtime container arguments')
        booleanParam(name: 'DRY_RUN_RUNTIME', defaultValue: true, description: 'Validate allocation and docker commands without launching real containers/models')
        choice(name: 'LOG_ARCHIVE_MODE', choices: ['auto', 'full', 'tail', 'failed-full', 'failed-tail', 'none'], description: 'Log archive policy. auto keeps full logs for nightly/release/benchmark and failed-case tails otherwise')
        string(name: 'LOG_ARCHIVE_TAIL_BYTES', defaultValue: '5242880', description: 'Max bytes retained per log in tail archive modes')
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
                stash name: 'ci-runtime-files', includes: '.ci/**/*,Jenkinsfile', allowEmpty: false
            }
        }

        stage('Precheck') {
            agent { label "${params.CPU_LABEL}" }
            steps {
                sh 'bash .ci/scripts/run_precheck.sh'
            }
        }

        stage('Run Unit Tests') {
            when { expression { return params.RUN_UT } }
            agent { label "${params.ASCEND_LABEL}" }
            steps {
                script {
                    runWithOptionalLock(params.ASCEND_LOCK_LABEL) {
                        sh '''#!/usr/bin/env bash
                            set -euo pipefail
                            mkdir -p reports/pytest
                            pytest_args=()
                            if [ -n "${UT_PYTEST_ARGS}" ]; then
                              pytest_args=(${UT_PYTEST_ARGS})
                            fi
                            marker_args=()
                            if [ -n "${UT_MARK_EXPR}" ]; then
                              marker_args=(-m "${UT_MARK_EXPR}")
                            fi
                            python3 -m pytest "${pytest_args[@]}" "${UT_TEST_PATH}" "${marker_args[@]}" \
                              --junitxml=reports/pytest/ut.xml
                        '''
                    }
                }
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'reports/pytest/ut.xml'
                    archiveArtifacts artifacts: 'reports/pytest/**/*', allowEmptyArchive: true
                    stash name: 'ut-reports', includes: 'reports/pytest/**/*', allowEmpty: true
                }
            }
        }

        stage('Validate Deploy Case Schema') {
            agent { label "${params.CPU_LABEL}" }
            steps {
                sh 'python3 .ci/scripts/ci.py validate --cases "${DEPLOY_CASE_GLOB}" --output reports/validated_cases.json'
                stash name: 'validated-report', includes: 'reports/validated_cases.json', allowEmpty: true
            }
        }

        stage('Generate Deploy Docs') {
            agent { label "${params.CPU_LABEL}" }
            steps {
                sh 'python3 .ci/scripts/ci.py render-docs --cases "${DEPLOY_CASE_GLOB}" --level all --output-dir docs/deploy/generated'
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
                        python3 .ci/scripts/ci.py select \
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
                    python3 .ci/scripts/ci.py static-validate \
                      --case-list reports/selected_cases.txt \
                      --model-root "${MODEL_ROOT}" \
                      --output reports/static_validate.json
                '''
                stash name: 'static-reports', includes: 'reports/selected_cases.txt,reports/static_validate.json', allowEmpty: true
            }
        }

        stage('Prepare Runtime Plan Inputs') {
            when { expression { return params.RUN_ASCEND } }
            agent { label "${params.CPU_LABEL}" }
            steps {
                unstash 'selected-cases'
                sh '''
                    mkdir -p reports/runtime_plan
                    cp reports/selected_cases.txt reports/runtime_plan/selected_cases.txt
                '''
                stash name: 'runtime-plan', includes: 'reports/runtime_plan/selected_cases.txt', allowEmpty: false
            }
        }

        stage('Run vLLM Ascend Deploy Cases') {
            when { expression { return params.RUN_ASCEND } }
            agent none
            steps {
                script {
                    def runtimeNodes = runtimeNodeCount(params)
                    if (!params.ASCEND_LOCK_LABEL?.trim()) {
                        error('RUN_ASCEND=true requires ASCEND_LOCK_LABEL so Jenkins can allocate Ascend node resources.')
                    }
                    lock(label: params.ASCEND_LOCK_LABEL, quantity: runtimeNodes, variable: 'LOCKED_ASCEND_NODES') {
                        echo "Acquired Ascend runtime resources: ${env.LOCKED_ASCEND_NODES}"
                        node(params.CPU_LABEL) {
                            unstash 'ci-runtime-files'
                            unstash 'selected-cases'
                            unstash 'runtime-plan'
                            def dockerImage = params.ASCEND_DOCKER_IMAGE?.trim()
                            if (!dockerImage) {
                                dockerImage = params.REGISTRY?.trim() ?
                                    "${params.REGISTRY}/vllm-ascend-ci:${params.IMAGE_TAG}" :
                                    "vllm-ascend-ci:${params.IMAGE_TAG}"
                            }
                            try {
                                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                                    withEnv([
                                        "ASCEND_DOCKER_IMAGE_RESOLVED=${dockerImage}",
                                        "ASCEND_DOCKER_DEVICE_ARGS=${params.ASCEND_DOCKER_DEVICE_ARGS}",
                                        "MODEL_ROOT=${params.MODEL_ROOT}",
                                        "DRY_RUN_RUNTIME=${params.DRY_RUN_RUNTIME}",
                                        "RUN_BENCHMARK_REQUESTED=${params.RUN_BENCHMARK}",
                                        "CI_MODE_RESOLVED=${params.CI_MODE}"
                                    ]) {
                                        sh '''#!/usr/bin/env bash
                                            set -euo pipefail
                                            mkdir -p reports/runtime_plan reports/nightly/case_results logs/deploy
                                            command -v ansible-playbook >/dev/null || {
                                              echo "ansible-playbook is required on the Jenkins controller/CPU agent for RUN_ASCEND=true"
                                              exit 1
                                            }

                                            python3 .ci/scripts/ci.py lock-inventory \
                                              --variable LOCKED_ASCEND_NODES \
                                              --output-json reports/runtime_plan/runtime_cluster_nodes.json \
                                              --output-inventory reports/runtime_plan/locked_ansible_inventory.yml

                                            dry_run_arg=()
                                            if [ "${DRY_RUN_RUNTIME}" = "true" ]; then
                                              dry_run_arg=(--dry-run)
                                            fi

                                            python3 .ci/scripts/ci.py compile-plan \
                                              --case-list reports/selected_cases.txt \
                                              --inventory-json reports/runtime_plan/runtime_cluster_nodes.json \
                                              --output-dir reports/runtime_plan \
                                              --model-root "${MODEL_ROOT}" \
                                              --docker-image "${ASCEND_DOCKER_IMAGE_RESOLVED}" \
                                              --host-workspace /home/ma-user/AscendCloud/jenkins \
                                              --extra-docker-args "${ASCEND_DOCKER_DEVICE_ARGS}" \
                                              "${dry_run_arg[@]}"

                                            ansible_args=()
                                            if [ "${DRY_RUN_RUNTIME}" = "true" ]; then
                                              ansible_args=(--check)
                                            fi

                                            ANSIBLE_CONFIG=.ci/ansible/ansible.cfg ansible-playbook \
                                              -i reports/runtime_plan/ansible_inventory.yml \
                                              .ci/ansible/playbooks/deploy_cases.yml \
                                              -e "dry_run_runtime=${DRY_RUN_RUNTIME}" \
                                              "${ansible_args[@]}"

                                            runtime_rc=0
                                            if [ "${DRY_RUN_RUNTIME}" = "true" ]; then
                                              python3 .ci/scripts/ci.py smoke \
                                                --case-list reports/selected_cases.txt \
                                                --deployment-plan reports/runtime_plan/deployment_plan.json \
                                                --output-dir reports/nightly/case_results \
                                                --model-root "${MODEL_ROOT}" \
                                                --continue-on-error \
                                                --dry-run \
                                                || runtime_rc=$?
                                            else
                                              ANSIBLE_CONFIG=.ci/ansible/ansible.cfg ansible-playbook \
                                                -i reports/runtime_plan/ansible_inventory.yml \
                                                .ci/ansible/playbooks/run_smoke.yml \
                                                -e "dry_run_runtime=false" \
                                                -e "model_root=${MODEL_ROOT}" \
                                                || runtime_rc=$?
                                            fi

                                            run_benchmark_runtime=false
                                            if [ "${RUN_BENCHMARK_REQUESTED}" = "true" ] \
                                              || [ "${CI_MODE_RESOLVED}" = "nightly" ] \
                                              || [ "${CI_MODE_RESOLVED}" = "benchmark" ] \
                                              || [ "${CI_MODE_RESOLVED}" = "release" ]; then
                                              run_benchmark_runtime=true
                                            fi
                                            if [ "${run_benchmark_runtime}" = "true" ]; then
                                              ANSIBLE_CONFIG=.ci/ansible/ansible.cfg ansible-playbook \
                                                -i reports/runtime_plan/ansible_inventory.yml \
                                                .ci/ansible/playbooks/run_benchmarks.yml \
                                                -e "dry_run_runtime=${DRY_RUN_RUNTIME}" \
                                                "${ansible_args[@]}" \
                                                || runtime_rc=$?
                                            fi
                                            exit "${runtime_rc}"
                                        '''
                                    }
                                }
                            } finally {
                                sh '''#!/usr/bin/env bash
                                    set +e
                                    if [ -f reports/runtime_plan/ansible_inventory.yml ]; then
                                      ANSIBLE_CONFIG=.ci/ansible/ansible.cfg ansible-playbook \
                                        -i reports/runtime_plan/ansible_inventory.yml \
                                        .ci/ansible/playbooks/cleanup_runtime.yml \
                                        -e "dry_run_runtime=${DRY_RUN_RUNTIME:-true}" \
                                        || echo "WARNING: failed to cleanup remote Ansible runtime"
                                      find reports/runtime_plan -name '*-remote-artifacts.tgz' -print \
                                        -exec tar -xzf {} -C . \\; \
                                        || echo "WARNING: failed to extract remote runtime artifacts"
                                    fi
                                    if [ -d reports/runtime_plan/benchmark_results ]; then
                                      python3 .ci/scripts/ci.py merge-benchmark \
                                        --case-results reports/nightly/case_results \
                                        --benchmark-results reports/runtime_plan/benchmark_results \
                                        || touch reports/runtime_plan/benchmark_failed
                                    fi
                                    bash .ci/scripts/cleanup_processes.sh logs/deploy
                                    python3 .ci/scripts/ci.py prepare-logs \
                                      --case-results reports/nightly/case_results \
                                      --logs-dir logs \
                                      --output-dir reports/logs \
                                      --manifest reports/nightly/log_artifacts.json \
                                      --mode "${LOG_ARCHIVE_MODE}" \
                                      --ci-mode "${CI_MODE}" \
                                      --tail-bytes "${LOG_ARCHIVE_TAIL_BYTES}" \
                                      || echo "WARNING: failed to prepare bounded log artifacts"
                                '''
                                def benchmarkFailed = fileExists('reports/runtime_plan/benchmark_failed')
                                stash name: 'runtime-results', includes: 'reports/nightly/case_results/**/*,reports/logs/**/*,reports/nightly/log_artifacts.json,reports/runtime_plan/**/*', allowEmpty: true
                                if (benchmarkFailed) {
                                    unstable('One or more DeployCase benchmark checks failed.')
                                }
                            }
                        }
                    }
                }
            }
        }

        stage('Run Benchmark') {
            when { expression { return params.RUN_BENCHMARK } }
            agent { label "${params.CPU_LABEL}" }
            steps {
                echo 'Benchmark execution is handled in the Ansible runtime stage when RUN_ASCEND=true.'
                echo 'The generated reports/nightly/benchmark.csv and case_results/*.json contain benchmark status and metrics.'
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
                    tryUnstash('ut-reports')
                }
                sh '''
                    python3 .ci/scripts/ci.py collect-env --output reports/nightly/environment.json
                    python3 .ci/scripts/ci.py junit \
                      --input reports/nightly/case_results \
                      --output reports/nightly/junit.xml
                    python3 .ci/scripts/ci.py report \
                      --input reports/nightly/case_results \
                      --environment reports/nightly/environment.json \
                      --output reports/nightly/index.html
                '''
                stash name: 'nightly-reports', includes: 'reports/**/*,docs/deploy/generated/**/*.md', allowEmpty: true
            }
        }

        stage('Archive reports and logs') {
            agent { label "${params.CPU_LABEL}" }
            steps {
                script {
                    tryUnstash('nightly-reports')
                }
                junit allowEmptyResults: true, testResults: 'reports/nightly/junit.xml,reports/pytest/*.xml'
                archiveArtifacts artifacts: 'reports/**/*,docs/deploy/generated/**/*.md', allowEmptyArchive: true
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
