pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'ghcr.io'
        IMAGE_NAME = 'ai-literature-review'
        GEMINI_API_KEY = credentials('gemini-api-key')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup Python') {
            steps {
                sh '''
                    python3 -m venv venv
                    source venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }
        
        stage('Run Tests') {
            steps {
                sh '''
                    source venv/bin/activate
                    pytest -q --tb=short
                '''
            }
            post {
                always {
                    publishTestResults testResultsPattern: 'test-results.xml'
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    def image = docker.build("${DOCKER_REGISTRY}/${env.IMAGE_NAME}:${env.BUILD_NUMBER}")
                    docker.image("${DOCKER_REGISTRY}/${env.IMAGE_NAME}:${env.BUILD_NUMBER}").push()
                    docker.image("${DOCKER_REGISTRY}/${env.IMAGE_NAME}:latest").push()
                }
            }
        }
        
        stage('Deploy to Render') {
            when {
                branch 'main'
            }
            steps {
                script {
                    // Trigger Render deployment via webhook
                    sh '''
                        curl -X POST "https://api.render.com/deploy/srv-d3inp9qli9vc73evub2g?key=--Ppy7v9yMM"
                    '''
                }
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
