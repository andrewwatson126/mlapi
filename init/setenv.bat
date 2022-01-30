set PATH=C:\Users\Cem Gültekin\bin\python-3.9.9-embed-amd64;%PATH%

pip3 install numpy
pip3 install pandas
pip3 install fastapi[all]
pip3 install uvicorn
pip3 install starlette
pip3 install matplotlib
pip3 install StringGenerator
pip3 install sklearn
pip3 install python-multipart
pip3 install aiofiles
pip3 install seaborn
pip3 install pytest
pip3 install requests

install vs code
 nodejs 
 python-3
 git
 
 uvicorn main:app --reload 
 uvicorn main:app --reload 
 uvicorn main:app --reload --log-config log.ini
 
 http://localhost:8000/docs
 
 
 ubuntu account
 nevilgultekin
 Rusher1290
 
# building docker images
docker build -t mlapi .
docker run -p 8000:8000 mlapi
docker run -p 8000:8000 mlapi 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlapi


install docker on aws
=============================
sudo yum update -y

sudo amazon-linux-extras install docker
sudo yum install docker

sudo service docker start



adding a docker image to aws ecr
================================
https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html

aws ecr create-repository --repository-name mlapi --region eu-central-1
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:eu-central-1:236653881753:repository/mlapi",
        "registryId": "236653881753",
        "repositoryName": "mlapi",
        "repositoryUri": "236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlapi",
        "createdAt": "2022-01-03T18:48:58+03:00",
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}

push docker image to ECR
====================
docker tag mlapi 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlapi
aws ecr get-login-password | docker login --username AWS --password-stdin 236653881753.dkr.ecr.eu-central-1.amazonaws.com
docker push 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlapi
aws ecr delete-repository --repository-name mlapi --region 236653881753 --force
docker pull 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlapi


root account id
796468841154 for nevil.gultekin@gmail.com

aws account id
236653881753

Administrator
Access key ID
AKIATOGNLVGMWHV7ZYHA

Secret access key
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

User name,Password,Access key ID,Secret access key,Console login link
Administrator,Rusher1290,AKIATOGNLVGMWHV7ZYHA,rn5LAxCAFCGhs+QKdW6SXNizHBDgYVXb3GP5RKJa,https://236653881753.signin.aws.amazon.com/console


              ecr.eu-central-1.amazonaws.com
 "https://api.ecr.eu-central.amazonaws.com/
          api.ecr.eu-central-1.amazonaws.com
 "https://api.ecr-public.eu-central-1.amazonaws.com/"



AWS Create EC2 with Linux 2 and docker
===================================================


https://medium.com/bb-tutorials-and-thoughts/running-docker-containers-on-aws-ec2-9b17add53646



EC2 Instance
VPC :
vpc-0a1ab3b04aadfc3f9	 Available	172.31.0.0/16

In EC2
sudo yum update -y
sudo amazon-linux-extras install docker
sudo service docker start
sudo usermod -a -G docker ec2-user


docker login -u AWS -p <password> <aws_account_id>.dkr.ecr.<region>.amazonaws.com



=================================================
MLAPI
=================================================

login to AWS
****************************
aws configure


Build and deploy docker image
***************************************
docker build -t mlapi .
docker run -p 8000:8000 mlapi
docker tag mlapi 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlapi
aws ecr get-login-password | docker login --username AWS --password-stdin 236653881753.dkr.ecr.eu-central-1.amazonaws.com
docker push 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlapi
aws ecr delete-repository --repository-name mlapi --region eu-central-1 --force



Steps on EC2
********************************************
sudo yum update -y
sudo amazon-linux-extras install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
aws ecr get-login-password | docker login --username AWS --password-stdin 236653881753.dkr.ecr.eu-central-1.amazonaws.com
docker pull 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlapi
docker run -p 8000:8000 mlapi 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlapi


=================================================
MLSTUDIO
=================================================
https://github.com/Schachte/simple-engineer-aws-deployment-tutorial

docker-compose -f docker-compose-development.yml build
docker run -p 8123:80 --add-host=apiserver:127.0.0.1 mlstudio_deployment-prod


docker tag mlstudio_deployment-prod 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlstudio_deployment-prod
aws ecr get-login-password | docker login --username AWS --password-stdin 236653881753.dkr.ecr.eu-central-1.amazonaws.com
docker push 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlstudio_deployment-prod
aws ecr delete-repository --repository-name mlstudio_deployment-prod --region eu-central-1 --force
EC2
aws ecr get-login-password | docker login --username AWS --password-stdin 236653881753.dkr.ecr.eu-central-1.amazonaws.com
docker pull 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlstudio_deployment-prod
docker run  -p 8000:8000 mlapi 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlstudio_deployment-prod




ubuntu 18 
username=nevilgultekin
password=Rusher1290





Echo  Build Images
Echo =================================
cd C:\Users\lenovo\projects\mlapi
docker build -t mlapi .
cd C:\Users\lenovo\projects\mlstudio
docker-compose -f docker-compose-development.yml build

Echo Push Tags
Echo =================================
docker tag mlapi 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlapi
aws ecr get-login-password | docker login --username AWS --password-stdin 236653881753.dkr.ecr.eu-central-1.amazonaws.com
docker push 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlapi

docker tag mlstudio_deployment-prod 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlstudio_deployment-prod
aws ecr get-login-password | docker login --username AWS --password-stdin 236653881753.dkr.ecr.eu-central-1.amazonaws.com
docker push 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlstudio_deployment-prod


AWS 
=====================================
sudo yum update -y
sudo amazon-linux-extras install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
aws ecr get-login-password | sudo docker login --username AWS --password-stdin 236653881753.dkr.ecr.eu-central-1.amazonaws.com
sudo docker pull 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlapi
sudo docker run -p 8000:8000 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlapi
sudo docker pull 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlstudio_deployment-prod
sudo docker run  -p 8000:8000 mlapi 236653881753.dkr.ecr.eu-central-1.amazonaws.com/mlstudio_deployment-prod
