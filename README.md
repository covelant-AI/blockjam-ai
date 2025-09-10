

## Models Used
https://huggingface.co/AmNat789/CovSports/tree/main


## Deployment



### Updating Network Volume on Runpod
https://docs.runpod.io/serverless/storage/network-volumes#architecture-details
https://docs.runpod.io/serverless/storage/s3-api

>#1 make a S3 API Key on Runpod > Settings > S3 Api Keys
- `$AccessKey`
- `$Secret`

>#2 Configure aws
```
aws configure


<!-- Fill the following info -->
AWS Access Key ID: [$AccessKey]
AWS Secret Access Key: [$Secret]
```


>#3 Connect with aws S3 API (Read Files in Volume)

```
aws s3 ls --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io s3://[NETWORK_VOLUME_ID]

<!-- Note:
    DATACENTER should be capitalized 
    NETWORK_VOLUME_ID is Storage > Bucket > Bucket Name
 -->
```


>#4 Upload files to Network Volume
```
aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io models/ s3://[NETWORK_VOLUME_ID]/ --recursive
```
