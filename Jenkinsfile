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

def ascendLabelForSoc(String soc, def pipelineParams) {
    if (soc == 'A2' && pipelineParams.ASCEND_A2_LABEL?.trim()) {
        return pipelineParams.ASCEND_A2_LABEL
    }
    if (soc == 'A3' && pipelineParams.ASCEND_A3_LABEL?.trim()) {
        return pipelineParams.ASCEND_A3_LABEL
    }
    return pipelineParams.ASCEND_LABEL
}

def sanitizeRuntimeName(String value) {
    return value.replaceAll(/[^A-Za-z0-9_.-]+/, '-').replaceAll(/^-+|-+$/, '')
}

def dockerMountSpecForShard(def shard, def pipelineParams, String workspacePath) {
    def lines = []
    for (def mount in (shard.docker?.mounts ?: [])) {
        def source = (mount.source ?: '').toString()
        def target = (mount.target ?: '').toString()
        def mode = (mount.mode ?: 'rw').toString()
        source = source.replace('${WORKSPACE}', workspacePath)
        source = source.replace('${MODEL_ROOT}', pipelineParams.MODEL_ROOT ?: '')
        target = target.replace('${MODEL_ROOT}', pipelineParams.MODEL_ROOT ?: '')
        if (source?.trim() && target?.trim()) {
            lines << "${source}\t${target}\t${mode}"
        }
    }
    return lines.join('\n')
}

def runtimeParallelism(def pipelineParams, int shardCount) {
    def configured = 0
    try {
        configured = pipelineParams.RUNTIME_PARALLELISM.toInteger()
    } catch (Exception ignored) {
        configured = 0
    }
    if (configured <= 0) {
        return Math.max(shardCount, 1)
    }
    return Math.max(configured, 1)
}

pipeline {
    agent none

    options {
        timestamps()
        ansiColor('xterm')
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
        string(name: 'ASCEND_A2_LABEL', defaultValue: '', description: 'Optional Jenkins label for 8-card A2 Ascend nodes')
        string(name: 'ASCEND_A3_LABEL', defaultValue: '', description: 'Optional Jenkins label for 16-card A3 Ascend nodes')
        string(name: 'ASCEND_LOCK_LABEL', defaultValue: '', description: 'Optional Lockable Resources label for Ascend hosts')
        string(name: 'ASCEND_DOCKER_IMAGE', defaultValue: '', description: 'Docker image used for Ascend runtime containers')
        string(name: 'ASCEND_DOCKER_DEVICE_ARGS', defaultValue: '', description: 'Device flags for docker run, for example --device entries or --privileged')
        string(name: 'NPU_LOCK_DIR', defaultValue: '/tmp/vllm-ascend-ci/npu', description: 'Host-local file lock directory for Ascend cards')
        string(name: 'PORT_LOCK_DIR', defaultValue: '/tmp/vllm-ascend-ci/ports', description: 'Host-local file lock directory for runtime ports')
        string(name: 'RUNTIME_PARALLELISM', defaultValue: '0', description: 'Max concurrent runtime shards; 0 means all planned shards')
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

        stage('Plan Runtime Shards') {
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
                    node(params.CPU_LABEL) {
                        unstash 'runtime-plan'
                        def plan = new JsonSlurperClassic().parseText(readFile('reports/runtime_shards.json'))
                        def shards = plan.shards ?: []
                        if (shards.isEmpty()) {
                            error('RUN_ASCEND=true but no runtime shards were planned.')
                        }

                        def maxParallel = runtimeParallelism(params, shards.size())
                        for (int start = 0; start < shards.size(); start += maxParallel) {
                            def end = Math.min(start + maxParallel, shards.size())
                            def batch = shards.subList(start, end)
                            def branches = [:]
                            for (def shard in batch) {
                                def localShard = shard
                                def shardName = localShard.name as String
                                branches[shardName] = {
                                    def label = ascendLabelForSoc(localShard.soc as String, params)
                                    node(label) {
                                        checkout scm
                                        unstash 'runtime-plan'
                                        def dockerImage = params.ASCEND_DOCKER_IMAGE?.trim()
                                        if (!dockerImage) {
                                            dockerImage = params.REGISTRY?.trim() ?
                                                "${params.REGISTRY}/vllm-ascend-ci:${params.IMAGE_TAG}" :
                                                "vllm-ascend-ci:${params.IMAGE_TAG}"
                                        }
                                        def containerName = sanitizeRuntimeName(
                                            "vllm-ascend-ci-${env.BUILD_TAG}-${localShard.case_name}"
                                        )
                                        def benchmarkArg = params.RUN_BENCHMARK ? '--run-benchmark' : ''
                                        def allocationJson = "reports/runtime_shards/${shardName}.allocation.json"
                                        def shardLogsDir = "logs/deploy/${shardName}"
                                        def dockerNetwork = localShard.docker?.network ?: 'host'
                                        def dockerIpc = localShard.docker?.ipc ?: 'host'
                                        def shmSize = localShard.docker?.shm_size ?: '64g'
                                        def dockerMountSpec = dockerMountSpecForShard(localShard, params, env.WORKSPACE)
                                        try {
                                            catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                                                withEnv([
                                                    "SHARD_CASE_LIST=${localShard.case_list}",
                                                    "SHARD_NAME=${shardName}",
                                                    "CARD_COUNT=${localShard.card_count}",
                                                    "PORT_COUNT=${localShard.port_count}",
                                                    "ALLOCATION_JSON=${allocationJson}",
                                                    "SHARD_LOGS_DIR=${shardLogsDir}",
                                                    "ASCEND_DOCKER_IMAGE_RESOLVED=${dockerImage}",
                                                    "ASCEND_DOCKER_DEVICE_ARGS=${params.ASCEND_DOCKER_DEVICE_ARGS}",
                                                    "CONTAINER_NAME=${containerName}",
                                                    "DOCKER_NETWORK=${dockerNetwork}",
                                                    "DOCKER_IPC=${dockerIpc}",
                                                    "SHM_SIZE=${shmSize}",
                                                    "DOCKER_MOUNT_SPEC=${dockerMountSpec}",
                                                    "MODEL_ROOT=${params.MODEL_ROOT}",
                                                    "NPU_LOCK_DIR=${params.NPU_LOCK_DIR}",
                                                    "PORT_LOCK_DIR=${params.PORT_LOCK_DIR}",
                                                    "BENCHMARK_ARG=${benchmarkArg}",
                                                    "DRY_RUN_RUNTIME=${params.DRY_RUN_RUNTIME}"
                                                ]) {
                                                    runWithOptionalLock(params.ASCEND_LOCK_LABEL) {
                                                        sh '''#!/usr/bin/env bash
                                                            set -euo pipefail
                                                            mkdir -p reports/nightly/case_results "${SHARD_LOGS_DIR}" reports/runtime_shards
                                                            benchmark_arg=()
                                                            if [ -n "${BENCHMARK_ARG}" ]; then
                                                              benchmark_arg=("${BENCHMARK_ARG}")
                                                            fi
                                                            mount_args=()
                                                            while IFS=$'\t' read -r mount_source mount_target mount_mode; do
                                                              if [ -n "${mount_source:-}" ] && [ -n "${mount_target:-}" ]; then
                                                                mount_args+=(-v "${mount_source}:${mount_target}:${mount_mode:-rw}")
                                                              fi
                                                            done <<< "${DOCKER_MOUNT_SPEC}"

                                                            if [ "${DRY_RUN_RUNTIME}" = "true" ]; then
                                                              python3 .ci/scripts/with_runtime_allocation.py \
                                                                --dry-run \
                                                                --card-count "${CARD_COUNT}" \
                                                                --port-count "${PORT_COUNT}" \
                                                                --npu-lock-dir "${NPU_LOCK_DIR}" \
                                                                --port-lock-dir "${PORT_LOCK_DIR}" \
                                                                --output "${ALLOCATION_JSON}"
                                                              python3 .ci/scripts/run_deploy_cases.py \
                                                                --case-list "${SHARD_CASE_LIST}" \
                                                                --allocation-json "${ALLOCATION_JSON}" \
                                                                --output-dir reports/nightly/case_results \
                                                                --logs-dir "${SHARD_LOGS_DIR}" \
                                                                --model-root "${MODEL_ROOT}" \
                                                                --dry-run \
                                                                --continue-on-error \
                                                                "${benchmark_arg[@]}"
                                                              echo "Docker dry-run for ${SHARD_NAME}:"
                                                              printf 'docker run --rm --name %q --network %q --ipc %q --shm-size %q %s -e ASCEND_RT_VISIBLE_DEVICES=<allocated>' "${CONTAINER_NAME}" "${DOCKER_NETWORK}" "${DOCKER_IPC}" "${SHM_SIZE}" "${ASCEND_DOCKER_DEVICE_ARGS}"
                                                              printf ' %q' "${mount_args[@]}"
                                                              printf ' -w /workspace/vllm-ascend %q python3 .ci/scripts/run_deploy_cases.py ...\n' "${ASCEND_DOCKER_IMAGE_RESOLVED}"
                                                            else
                                                              python3 .ci/scripts/with_runtime_allocation.py \
                                                                --card-count "${CARD_COUNT}" \
                                                                --port-count "${PORT_COUNT}" \
                                                                --npu-lock-dir "${NPU_LOCK_DIR}" \
                                                                --port-lock-dir "${PORT_LOCK_DIR}" \
                                                                --output "${ALLOCATION_JSON}" \
                                                                -- bash -lc '
                                                                  set -euo pipefail
                                                                  benchmark_arg=()
                                                                  if [ -n "${BENCHMARK_ARG}" ]; then
                                                                    benchmark_arg=("${BENCHMARK_ARG}")
                                                                  fi
                                                                  model_env_args=()
                                                                  if [ -n "${MODEL_ROOT}" ]; then
                                                                    model_env_args=(-e MODEL_ROOT="${MODEL_ROOT}")
                                                                  fi
                                                                  mount_args=()
                                                                  while IFS=$'\t' read -r mount_source mount_target mount_mode; do
                                                                    if [ -n "${mount_source:-}" ] && [ -n "${mount_target:-}" ]; then
                                                                      mount_args+=(-v "${mount_source}:${mount_target}:${mount_mode:-rw}")
                                                                    fi
                                                                  done <<< "${DOCKER_MOUNT_SPEC}"
                                                                  docker run --rm \
                                                                    --name "${CONTAINER_NAME}" \
                                                                    --network "${DOCKER_NETWORK}" \
                                                                    --ipc "${DOCKER_IPC}" \
                                                                    --shm-size "${SHM_SIZE}" \
                                                                    ${ASCEND_DOCKER_DEVICE_ARGS} \
                                                                    -e ASCEND_RT_VISIBLE_DEVICES \
                                                                    -e VLLM_CI_ALLOCATED_PORTS \
                                                                    -e VLLM_CI_ALLOCATION_JSON="${ALLOCATION_JSON}" \
                                                                    -e VLLM_CI_CONTAINER_NAME="${CONTAINER_NAME}" \
                                                                    "${model_env_args[@]}" \
                                                                    "${mount_args[@]}" \
                                                                    -w /workspace/vllm-ascend \
                                                                    "${ASCEND_DOCKER_IMAGE_RESOLVED}" \
                                                                    python3 .ci/scripts/run_deploy_cases.py \
                                                                      --case-list "${SHARD_CASE_LIST}" \
                                                                      --allocation-json "${ALLOCATION_JSON}" \
                                                                      --output-dir reports/nightly/case_results \
                                                                      --logs-dir "${SHARD_LOGS_DIR}" \
                                                                      --model-root "${MODEL_ROOT}" \
                                                                      --continue-on-error \
                                                                      "${benchmark_arg[@]}"
                                                                '
                                                            fi
                                                        '''
                                                    }
                                                }
                                            }
                                        } finally {
                                            sh 'bash .ci/scripts/cleanup_processes.sh logs/deploy || true'
                                            stash name: "runtime-results-${shardName}", includes: 'reports/nightly/case_results/**/*,logs/**/*,reports/runtime_shards/*.allocation.json', allowEmpty: true
                                        }
                                    }
                                }
                            }
                            parallel branches
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
                    if (fileExists('reports/runtime_shards.json')) {
                        def plan = new JsonSlurperClassic().parseText(readFile('reports/runtime_shards.json'))
                        for (def shard in (plan.shards ?: [])) {
                            tryUnstash("runtime-results-${shard.name}")
                        }
                    } else {
                        tryUnstash('runtime-results')
                    }
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
