terraform {}

provider "aws" {
 region = "us-west-2"
}

resource "aws_s3_bucket" "shuga" {
 bucket = var.bucket_name
 versioning {
 enabled = true
 }
 server_side_encryption_configuration {
 rule {
 apply_server_side_encryption_by_default {
 sse_algorithm = "AES256"
 }
 }
 }
 lifecycle_rule {
 prefix = "images/"
 enabled = true
 transition {
 days = 30
 storage_class = "STANDARD_IA"
 }
}
}