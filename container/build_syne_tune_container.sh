#!/usr/bin/env bash

# allows to build a docker image that can be used to run Syne Tune.
set -x

CONTEXT="cpu-py36"

rm -rf ./source && mkdir ./source
cat ../requirements.txt >> ./source/requirements.txt
cat ../requirements-gpsearchers.txt >> ./source/requirements.txt
cat ../requirements-ray.txt >> ./source/requirements.txt

echo "installing the following dependencies in the docker image"
echo `cat ./source/requirements.txt`

# Name passed to the command
image="syne-tune-$CONTEXT"

# Get the account number associated with the current IAM credentials
# this will be a 12 digit number (the main AWS); assumes you've done
# aws configure etc.
account=$(aws sts get-caller-identity --query Account --output text)

# in case the exit status of previous task is not 0 (success), exit
# (e.g. badly set AWS credentials)
if [ $? -ne 0 ]
then
    exit 255
fi

# Get the region defined in the current configuration (default to us-west-2 if none)
region=$(aws configure get region)
region=${region:-us-west-2}

# something like xxxxxxxxxxxx.dkr.ecr.us-west-2.amazonaws.com
ECR_URL="${account}.dkr.ecr.${region}.amazonaws.com"

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

DLAMI_REGISTRY_ID=763104351884
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin $DLAMI_REGISTRY_ID.dkr.ecr.${region}.amazonaws.com


# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# build the image
# Note, `.` indicates it will take the dockerfile available here.
# the build args are useful for the Deep Learning Container setup
docker build -t ${image} . \
             --build-arg REGION=${region} \
             --build-arg DLAMI_REGISTRY_ID=${DLAMI_REGISTRY_ID} \
             --build-arg CONTEXT=${CONTEXT} \
             --pull \
             --no-cache

# tag and push to ECR
docker tag ${image} ${ECR_URL}/${image}
docker push ${ECR_URL}/${image}
