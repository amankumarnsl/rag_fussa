"""
Simple script to upload sample_data.pdf to S3
"""
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def upload_pdf_to_s3():
    """Upload sample_data.pdf to S3 bucket"""
    
    # Get AWS credentials from environment
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    
    # You'll need to set this in your .env file
    bucket_name = os.getenv("AWS_S3_BUCKET")
    
    if not all([aws_access_key_id, aws_secret_access_key, bucket_name]):
        print("❌ Missing AWS credentials or bucket name in .env file")
        print("Please set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET")
        return None
    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    
    # # File details
    # local_file = "sample_data.pdf"
    # s3_key = "documents/sample_data.pdf"  # Path in S3 bucket

    # # Video file
    # local_file = "sample_data.mp4"
    # s3_key = "documents/sample_data.mp4"  # Path in S3 bucket

    # Video file working file structure
    local_file = "data_to_upload_to_S3/Handbook on Legal System & Procedure.pdf"
    s3_key = "documents/Handbook on Legal System & Procedure.pdf"  # Path in S3 bucket
    
    try:
        # Check if file exists locally
        if not os.path.exists(local_file):
            print(f"❌ File {local_file} not found in current directory")
            return None
        
        # Upload file
        print(f"📤 Uploading {local_file} to s3://{bucket_name}/{s3_key}...")
        
        s3_client.upload_file(
            local_file, 
            bucket_name, 
            s3_key,
            ExtraArgs={'ContentType': 'application/pdf'}
        )
        
        # Generate S3 URL
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        
        print(f"✅ Successfully uploaded!")
        print(f"📄 S3 URL: {s3_url}")
        print(f"\n🚀 Now you can use this URL to train:")
        print(f'curl -X POST "http://localhost:8000/train" \\')
        print(f'     -H "Content-Type: application/json" \\')
        print(f'     -d \'{{"s3_url": "{s3_url}"}}\'')
        
        return s3_url
        
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        return None


if __name__ == "__main__":
    upload_pdf_to_s3()
