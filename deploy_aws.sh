#!/bin/bash
# AWS Deployment Script for MedGemma Healthcare AI Platform

echo "🚀 Starting AWS Deployment for MedGemma Healthcare AI Platform"

# Step 1: Build Docker image
echo "📦 Building Docker image..."
docker build -t medgemma-healthcare-ai .

# Step 2: Tag for ECR (replace with your AWS account ID and region)
echo "🏷️ Tagging image for AWS ECR..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/medgemma-healthcare-ai"

docker tag medgemma-healthcare-ai:latest $ECR_URI:latest

# Step 3: Login to ECR
echo "🔐 Logging into AWS ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI

# Step 4: Create ECR repository if it doesn't exist
echo "📂 Creating ECR repository..."
aws ecr create-repository --repository-name medgemma-healthcare-ai --region $AWS_REGION || true

# Step 5: Push image to ECR
echo "⬆️ Pushing image to ECR..."
docker push $ECR_URI:latest

echo "✅ Docker image pushed to ECR: $ECR_URI:latest"

# Step 6: Deploy to ECS (optional)
echo "🌐 Ready for ECS deployment!"
echo "Next steps:"
echo "1. Create ECS cluster: aws ecs create-cluster --cluster-name medgemma-cluster"
echo "2. Create task definition using the ECR image URI: $ECR_URI:latest"
echo "3. Create ECS service with load balancer"
echo "4. Set environment variables: XAI_API_KEY, OPENAI_API_KEY"

echo "🎉 AWS Deployment preparation complete!"