import boto3, os
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url="https://s3.ru-7.storage.selcloud.ru",
    aws_access_key_id=os.environ["KOZ3_S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["KOZ3_S3_SECRET_KEY"],
    region_name="ru-7", config=Config(s3={"addressing_style":"path"}))
n=0
for l in open("docs/s3_listing.tsv",encoding="utf-8"):
    s,k=l.rstrip("\n").split("\t",1)
    if k.lower().endswith((".geojson",".json")):
        out=os.path.join("data_meta",k); os.makedirs(os.path.dirname(out),exist_ok=True)
        if not os.path.exists(out): s3.download_file("train-expds-3",k,out); n+=1
print("downloaded",n)
