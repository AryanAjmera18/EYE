import os
import subprocess 

import subprocess
import os

class S3Sync:
    def sync_folder_to_s3(self, folder, aws_bucket_url): 
        print(f"\n📁 Syncing folder: {folder}")
        print(f"☁️  Destination: {aws_bucket_url}")
        print(f"📂 Folder exists: {os.path.exists(folder)}")
        print(f"📦 Files in folder:")
        
        for root, dirs, files in os.walk(folder):
            for file in files:
                print("   -", os.path.join(root, file))

        command = f'aws s3 sync "{folder}" "{aws_bucket_url}" --exact-timestamps'
        print(f"🚀 Running command: {command}")

        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Error occurred while syncing:")
            print(result.stderr)
        else:
            print("✅ Upload completed successfully.")
            print(result.stdout)

    
    def sync_folder_from_s3(self,folder,aws_bucket_url):
        command = f"aws s3 sync {aws_bucket_url}{folder}"
        os.system(command)    
            