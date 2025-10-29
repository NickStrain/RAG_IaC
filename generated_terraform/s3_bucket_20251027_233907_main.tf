{
    "main_tf": """
resource "aws_s3_bucket" "shuga" {
  bucket = var.bucket_name
  acl    = "private"

  versioning {
    enabled = var.versioning
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}
""",
    "variables_tf": """
variable "bucket_name" {
  description = "The name of the S3 bucket."
  type        = string
}

variable "aws_region" {
  description = "The AWS region to deploy the bucket in."
  type        = string
}

variable "versioning" {
  description = "Whether versioning is enabled for the bucket."
  type        = bool
  default     = true
}

variable "encryption" {
  description = "Whether encryption is enabled for the bucket."
  type        = bool
  default     = true
}""",
    "outputs_tf": """
output "bucket_name" {
  value       = aws_s3_bucket.shuga.id
  description = "The name of the S3 bucket."
}

output "versioning_enabled" {
  value       = aws_s3_bucket.shuga.versioning.0.enabled
  description = "Whether versioning is enabled for the bucket."
}""",
    "terraform_tfvars": """
bucket_name  = "shuga"
aws_region   = "us-east"
versioning   = true
encryption   = true"""
}