# CamVid data download
cd /workspace/pspnet-mlops

git clone https://github.com/alexgkendall/SegNet-Tutorial.git /tmp/segnet
cp -r /tmp/segnet/CamVid data/camvid
rm -rf /tmp/segnet

# check data files
ls data/camvid/

# MLFlow 
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --allowed-hosts "*"