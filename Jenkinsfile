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
        string(name: 'ASCEND_LOCK_LABEL', defaultValue: '', description: 'Optional Lockable Resources label for Ascend hosts')
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

        stage('Run vLLM Ascend Deploy Cases') {
            when { expression { return params.RUN_ASCEND } }
            agent { label "${params.ASCEND_LABEL}" }
            steps {
                script {
                    unstash 'selected-cases'
                    def benchmarkArg = params.RUN_BENCHMARK ? '--run-benchmark' : ''
                    def command = """
                        python3 .ci/scripts/run_deploy_cases.py \
                          --case-list reports/selected_cases.txt \
                          --output-dir reports/nightly/case_results \
                          --logs-dir logs/deploy \
                          --model-root "${params.MODEL_ROOT}" \
                          --continue-on-error \
                          ${benchmarkArg}
                    """
                    catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                        runWithOptionalLock(params.ASCEND_LOCK_LABEL) {
                            sh command
                        }
                    }
                }
            }
            post {
                always {
                    sh 'bash .ci/scripts/cleanup_processes.sh logs/deploy || true'
                    stash name: 'runtime-results', includes: 'reports/nightly/case_results/**/*,logs/**/*', allowEmpty: true
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
