This directory includes a Docker image necessary to run `syne_tune.remote.remote_launcher.RemoteLauncher`.

The first time you use `RemoteLauncher` this image will be built automatically and uploaded/pushed to your AWS ECR (Elastic Container Registry).

It might happen that for a once-built image, with time some dependencies will grow out of date. 
For example, if not updated, AWS Python libraries like `botocore` or `boto3` might not offer the newest features of the 
AWS API as [documented online](https://docs.aws.amazon.com/sagemaker/latest/APIReference/).
In that case you need to rebuild the image, installing the most recent versions of all of the dependencies.
To do that, run `build_syne_tune_container.sh` from within the `container/` directory:
```
cd container/
bash build_syne_tune_container.sh
```
