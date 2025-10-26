# providers.tf
# Define the AWS provider and region
provider "aws" {
  region = var.aws_region
}

# main.tf
# Data source to retrieve the current AWS account ID, used in IAM policies.
data "aws_caller_identity" "current" {}

# Resource to generate a random suffix for bucket names, ensuring global uniqueness
# when a bucket_name is not explicitly provided by the user.
resource "random_id" "bucket_suffix" {
  byte_length = 8
}

# Local values for bucket names, allowing for dynamic default uniqueness.
locals {
  # If 'var.bucket_name' is explicitly set, use it.
  # Otherwise, generate a unique name using a prefix and a random suffix.
  final_main_bucket_name = coalesce(var.bucket_name, "my-app-data-${random_id.bucket_suffix.hex}")
  # The log bucket name is derived from the main bucket name.
  final_log_bucket_name  = "${local.final_main_bucket_name}${var.log_bucket_suffix}"
}

# AWS Key Management Service (KMS) Key for S3 Server-Side Encryption (SSE-KMS)
# This key provides more control over encryption keys compared to S3-managed keys (SSE-S3).
resource "aws_kms_key" "s3_bucket_kms_key" {
  description             = "KMS key for S3 bucket ${local.final_main_bucket_name} encryption"
  deletion_window_in_days = 10 # Required for KMS key deletion to prevent immediate accidental deletion
  enable_key_rotation     = true # Automatically rotate the key annually

  # KMS Key Policy allows the AWS account to delegate permissions via IAM policies,
  # and allows the S3 service to use the key for encryption and decryption.
  policy = jsonencode({
    Version = "2012-10-17"
    Id      = "key-policy-1"
    Statement = [
      {
        Sid       = "Enable IAM User Permissions"
        Effect    = "Allow"
        # Grants the AWS account full access to the KMS key, allowing IAM policies
        # to grant specific permissions to users/roles within the account.
        Principal = { AWS = data.aws_caller_identity.current.account_id }
        Action    = "kms:*"
        Resource  = "*"
      },
      {
        Sid       = "Allow S3 to use the key for encryption and decryption"
        Effect    = "Allow"
        Principal = { Service = "s3.amazonaws.com" }
        Action = [
          "kms:GenerateDataKey",
          "kms:Decrypt"
        ]
        Resource = "*"
      }
    ]
  })

  tags = {
    Name        = "${local.final_main_bucket_name}-kms-key"
    Environment = var.environment
    Project     = var.project
    ManagedBy   = "Terraform"
  }
}

# KMS Key Alias for easier referencing of the KMS key
resource "aws_kms_alias" "s3_bucket_kms_alias" {
  name          = "alias/${var.kms_key_alias_name}"
  target_key_id = aws_kms_key.s3_bucket_kms_key.key_id
}

# S3 Log Bucket Resource
# This dedicated bucket stores server access logs for the main S3 bucket.
# It uses "BucketOwnerEnforced" for object ownership, which is a best practice for logging buckets.
# This ensures logs are owned by the bucket owner and access is managed purely through bucket policies,
# making ACLs irrelevant for this bucket.
resource "aws_s3_bucket" "log_bucket" {
  bucket = local.final_log_bucket_name

  # "BucketOwnerEnforced" disables ACLs; all access control is via bucket policies.
  # This is the recommended best practice for new buckets.
  object_ownership = "BucketOwnerEnforced"

  # Prevent accidental deletion of this critical log bucket.
  lifecycle {
    prevent_destroy = true
  }

  tags = {
    Name        = "${local.final_main_bucket_name}-logs"
    Environment = var.environment
    Project     = var.project
    ManagedBy   = "Terraform"
  }
}

# S3 Log Bucket Public Access Block (Essential for security)
# This resource prevents any form of public access to the log bucket, ensuring logs remain private.
resource "aws_s3_bucket_public_access_block" "log_bucket_public_access_block" {
  bucket = aws_s3_bucket.log_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Log Bucket Versioning
# Enables versioning for the log bucket to protect against accidental deletion or overwrites of log files.
resource "aws_s3_bucket_versioning" "log_bucket_versioning" {
  bucket = aws_s3_bucket.log_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Log Bucket Server-Side Encryption Configuration (SSE-S3)
# Enforces server-side encryption for all log objects stored in this bucket using S3-managed keys (AES256).
# This is generally sufficient for log data.
resource "aws_s3_bucket_server_side_encryption_configuration" "log_bucket_encryption" {
  bucket = aws_s3_bucket.log_bucket.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 Log Bucket Lifecycle Configuration
# Defines rules to manage the lifecycle of log objects, such as expiration, to optimize storage costs.
resource "aws_s3_bucket_lifecycle_configuration" "log_bucket_lifecycle" {
  bucket = aws_s3_bucket.log_bucket.id

  rule {
    id     = "expire-logs"
    status = "Enabled"

    expiration {
      days = var.log_bucket_lifecycle_expiration_days # Logs expire after a specified number of days
    }

    # For versioned buckets, it's good practice to also expire non-current versions
    noncurrent_version_expiration {
      noncurrent_days = var.log_bucket_lifecycle_expiration_days # Expire non-current log versions
    }
  }
}

# S3 Log Bucket Policy
# This policy allows the S3 service principal to deliver access logs to this bucket.
# It is necessary because `object_ownership` is set to "BucketOwnerEnforced" for this logging bucket,
# which relies solely on bucket policies for access control.
resource "aws_s3_bucket_policy" "log_bucket_policy" {
  bucket = aws_s3_bucket.log_bucket.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "S3BucketPolicyForS3Logging"
        Effect    = "Allow"
        Principal = { Service = "logging.s3.amazonaws.com" }
        Action    = "s3:PutObject"
        Resource  = "${aws_s3_bucket.log_bucket.arn}/*"
      }
    ]
  })
}

# Main S3 Bucket Resource
# This is the primary S3 bucket for application data.
resource "aws_s3_bucket" "main_bucket" {
  bucket = local.final_main_bucket_name

  # "BucketOwnerEnforced" is the recommended best practice. It simplifies
  # permission management by disabling ACLs entirely, relying solely on
  # IAM and bucket policies. This aligns with modern S3 best practices.
  object_ownership = "BucketOwnerEnforced"

  # Prevent accidental deletion of this critical production bucket.
  lifecycle {
    prevent_destroy = true
  }

  tags = {
    Name        = local.final_main_bucket_name
    Environment = var.environment
    Project     = var.project
    ManagedBy   = "Terraform"
  }
}

# The 'aws_s3_bucket_acl' resource is removed because 'object_ownership = "BucketOwnerEnforced"'
# disables ACLs and relies purely on IAM and bucket policies for access control,
# which is the recommended best practice.

# S3 Bucket Public Access Block (Crucial for security)
# This resource prevents public access to the bucket by blocking public ACLs,
# public policies, and overriding public access granted inadvertently.
resource "aws_s3_bucket_public_access_block" "main_bucket_public_access_block" {
  bucket = aws_s3_bucket.main_bucket.id

  block_public_acls       = true # Blocks new public ACLs and updating existing public ACLs
  block_public_policy     = true # Blocks new public bucket policies and updating existing public bucket policies
  ignore_public_acls      = true # Ignores public ACLs on objects when evaluating object access
  restrict_public_buckets = true # Restricts access to only AWS services and authorized users if a bucket has a public policy
}

# S3 Bucket Versioning Configuration
# Enables versioning to keep multiple variants of an object, allowing recovery from
# accidental deletions or overwrites.
resource "aws_s3_bucket_versioning" "main_bucket_versioning" {
  bucket = aws_s3_bucket.main_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket Server-Side Encryption Configuration (SSE-KMS)
# Enforces server-side encryption for all objects stored in the main bucket using the
# AWS KMS key defined above, providing more granular control over encryption keys.
resource "aws_s3_bucket_server_side_encryption_configuration" "main_bucket_encryption" {
  bucket = aws_s3_bucket.main_bucket.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3_bucket_kms_key.arn # Use the ARN of the created KMS key
    }
  }
}

# S3 Bucket Logging Configuration
# Enables server access logging for the main bucket, sending logs to the dedicated log bucket.
# This is crucial for auditing and security monitoring.
resource "aws_s3_bucket_logging" "main_bucket_logging" {
  bucket        = aws_s3_bucket.main_bucket.id
  target_bucket = aws_s3_bucket.log_bucket.id
  target_prefix = "s3-access-logs/" # Prefix to organize logs within the target log bucket
}

# S3 Bucket Lifecycle Configuration
# Defines rules for object transitions and expiration in the main bucket, optimizing storage costs.
resource "aws_s3_bucket_lifecycle_configuration" "main_bucket_lifecycle" {
  bucket = aws_s3_bucket.main_bucket.id

  rule {
    id     = "standard-to-ia-to-glacier-and-expire"
    status = "Enabled"

    # Transition objects to Standard-IA (Infrequent Access) after a specified period
    transition {
      days          = var.main_bucket_lifecycle_transition_ia_days
      storage_class = "STANDARD_IA"
    }

    # Transition objects to Glacier after another specified period for archival
    transition {
      days          = var.main_bucket_lifecycle_transition_glacier_days
      storage_class = "GLACIER"
    }

    # Expire current versions of objects after a very long period (e.g., 10 years)
    expiration {
      days = var.main_bucket_lifecycle_expiration_days_current
    }

    # Expire non-current versions of objects after a shorter period (e.g., 7 days)
    # This helps manage costs while retaining versioning for recent changes.
    noncurrent_version_expiration {
      noncurrent_days = var.main_bucket_lifecycle_expiration_days_noncurrent
    }
  }
}


# outputs.tf
# Outputs provide critical attributes of the created resources for external reference.

output "main_bucket_id" {
  description = "The ID (name) of the main S3 bucket."
  value       = aws_s3_bucket.main_bucket.id
}

output "main_bucket_arn" {
  description = "The ARN of the main S3 bucket."
  value       = aws_s3_bucket.main_bucket.arn
}

output "main_bucket_regional_domain_name" {
  description = "The regional domain name of the main S3 bucket."
  value       = aws_s3_bucket.main_bucket.bucket_regional_domain_name
}

output "log_bucket_id" {
  description = "The ID (name) of the S3 log bucket."
  value       = aws_s3_bucket.log_bucket.id
}

output "log_bucket_arn" {
  description = "The ARN of the S3 log bucket."
  value       = aws_s3_bucket.log_bucket.arn
}

output "log_bucket_regional_domain_name" {
  description = "The regional domain name of the S3 log bucket."
  value       = aws_s3_bucket.log_bucket.bucket_regional_domain_name
}

output "s3_kms_key_arn" {
  description = "The ARN of the KMS key used for S3 encryption."
  value       = aws_kms_key.s3_bucket_kms_key.arn
}

output "s3_kms_key_alias_arn" {
  description = "The ARN of the KMS key alias used for S3 encryption."
  value       = aws_kms_alias.s3_bucket_kms_alias.arn
}


# variables.tf
# Input variables for the S3 bucket and associated resource configurations.

variable "bucket_name" {
  description = "The name for the main S3 bucket. If not provided, a unique name will be generated using a default prefix and a random suffix."
  type        = string
  default     = null # Setting default to null allows 'coalesce' in locals to provide a dynamic unique name.
}

variable "aws_region" {
  description = "The AWS region where resources will be created."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "The deployment environment (e.g., 'prod', 'staging', 'dev'). Used for tagging."
  type        = string
  default     = "prod"
}

variable "project" {
  description = "The project name this bucket belongs to. Used for tagging."
  type        = string
  default     = "MyWebApp"
}

variable "kms_key_alias_name" {
  description = "The alias name for the KMS key, which will be prefixed with 'alias/'."
  type        = string
  default     = "s3/main-app-bucket-key"
}

variable "log_bucket_suffix" {
  description = "A suffix to append to the main bucket name to create the unique log bucket name."
  type        = string
  default     = "-logs"
}

variable "main_bucket_lifecycle_expiration_days_current" {
  description = "Number of days after object creation to expire current versions in the main bucket. Set to a very long period for important data."
  type        = number
  default     = 3650 # 10 years
}

variable "main_bucket_lifecycle_transition_ia_days" {
  description = "Number of days after object creation to transition to STANDARD_IA storage class in the main bucket."
  type        = number
  default     = 30
}

variable "main_bucket_lifecycle_transition_glacier_days" {
  description = "Number of days after object creation to transition to GLACIER storage class in the main bucket."
  type        = number
  default     = 90
}

variable "main_bucket_lifecycle_expiration_days_noncurrent" {
  description = "Number of days after an object becomes a non-current version to expire it in the main bucket."
  type        = number
  default     = 7 # Keep non-current versions for 7 days
}

variable "log_bucket_lifecycle_expiration_days" {
  description = "Number of days after object creation to expire all objects (logs) in the log bucket."
  type        = number
  default     = 365 # 1 year
}