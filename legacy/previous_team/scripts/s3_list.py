import boto3, sys, collections, json, os
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url="https://s3.ru-7.storage.selcloud.ru",
    aws_access_key_id=os.environ["KOZ3_S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["KOZ3_S3_SECRET_KEY"],
    region_name="ru-7", config=Config(s3={"addressing_style":"path"}))
rows=[]
for page in s3.get_paginator("list_objects_v2").paginate(Bucket="train-expds-3"):
    for o in page.get("Contents",[]): rows.append((o["Key"],o["Size"]))
out=sys.argv[1]
with open(out,"w",encoding="utf-8") as f:
    for k,s in rows: f.write(f"{s}\t{k}\n")
print("objects",len(rows),"total GB",round(sum(s for _,s in rows)/1e9,2))
