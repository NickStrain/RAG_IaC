# Configure the AWS provider
provider "aws" {
  region = var.aws_region # Use the externalized AWS region variable
}

# Input variables for customization
variable "aws_region" {
  description = "The AWS region where resources will be deployed."
  type        = string
  default     = "us-east-1"
}

variable "bucket_name_prefix" {
  description = "A unique prefix for the S3 bucket name. A random suffix will be appended."
  type        = string
  default     = "my-secure-app-bucket"
}

variable "logging_bucket_name_prefix" {
  description = "A unique prefix for the S3 logging bucket name. A random suffix will be appended."
  type        = string
  default     = "my-app-logs-bucket"
}

variable "environment" {
  description = "The deployment environment (e.g., dev, stage, prod)."
  type        = string
  default     = "production"
}

variable "application_iam_role_arn" {
  description = "ARN of the IAM role that needs access to the main S3 bucket for application operations. Example: arn:aws:iam::123456789012:role/MyApplicationRole"
  type        = string
  default     = "" # Provide a specific IAM role ARN for application access or leave empty if not immediately known
}

# Data source to get the current AWS account ID dynamically
data "aws_caller_identity" "current" {}

# Generate random strings to ensure S3 bucket names are globally unique
resource "random_id" "bucket_suffix" {
  byte_length = 8
}

resource "random_id" "logging_bucket_suffix" {
  byte_length = 8
}

# -----------------------------------------------------------------------------
# AWS KMS Key for Server-Side Encryption
# -----------------------------------------------------------------------------
# Create an AWS KMS Key dedicated for S3 server-side encryption.
# Using KMS provides greater control over encryption keys, allows for auditing
# key usage, and supports automatic key rotation, enhancing security beyond SSE-S3 (AES256).
resource "aws_kms_key" "s3_encryption_key" {
  description             = "KMS key for S3 bucket ${var.bucket_name_prefix}"
  deletion_window_in_days = 7 # Minimum value is 7 days, allows time to recover from accidental deletion
  enable_key_rotation     = true # Automatically rotate the key annually for enhanced security

  # Define the key policy to allow the account root user and the S3 service
  # to use the key for encryption/decryption. This is crucial for S3 default encryption.
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        # Allow the account root user to administer the key
        Principal = { AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action    = "kms:*"
        Resource  = "*"
      },
      {
        Sid       = "Allow S3 to use the key for bucket default encryption"
        Effect    = "Allow"
        Principal = { Service = "s3.amazonaws.com" }
        Action    = [
          "kms:GenerateDataKey",
          "kms:Decrypt",
          "kms:Encrypt"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            # Restrict key usage to operations originating from the specific AWS account.
            # The bucket's default encryption configuration implicitly links the key to the bucket.
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      },
      # Dynamically add a statement to allow a specific application IAM role to use the KMS key
      # if 'application_iam_role_arn' is provided. This grants necessary permissions
      # for the application to interact with encrypted objects.
      dynamic "statement" {
        for_each = var.application_iam_role_arn != "" ? [1] : []
        content {
          Sid       = "AllowApplicationRoleToUseKMSKey"
          Effect    = "Allow"
          Principal = { AWS = var.application_iam_role_arn }
          Action    = [
            "kms:Decrypt",
            "kms:Encrypt",
            "kms:ReEncrypt*",
            "kms:GenerateDataKey*",
            "kms:DescribeKey"
          ]
          Resource = "*"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.bucket_name_prefix}-kms-key"
    Environment = var.environment
    ManagedBy   = "Terraform"
    Purpose     = "S3Encryption"
  }
}

# Add a KMS Key Alias for improved manageability and easier referencing in the AWS Console.
resource "aws_kms_alias" "s3_encryption_alias" {
  name          = "alias/${var.bucket_name_prefix}-s3-key"
  target_key_id = aws_kms_key.s3_encryption_key.id

  tags = {
    Name        = "${var.bucket_name_prefix}-kms-key-alias"
    Environment = var.environment
    ManagedBy   = "Terraform"
    Purpose     = "S3EncryptionAlias"
  }
}

# -----------------------------------------------------------------------------
# Main Application S3 Bucket
# (Defined before logging bucket policy to resolve forward reference)
# -----------------------------------------------------------------------------
# Create an S3 bucket for application data with versioning and server-side encryption (KMS).
resource "aws_s3_bucket" "production_bucket" {
  # Construct a globally unique bucket name using a prefix and a random suffix
  bucket = "${var.bucket_name_prefix}-${random_id.bucket_suffix.hex}"

  # Enable S3 Versioning to keep multiple variants of an object.
  # This helps in preserving, retrieving, and restoring every version of objects,
  # recovering from unintended user actions or application failures.
  versioning {
    enabled = true
    # For extremely sensitive data, consider enabling MFA Delete here:
    # mfa_delete = "Enabled"
    # IMPORTANT: Enabling MFA Delete requires an MFA device serial number,
    # which is not typically managed declaratively in Terraform for initial setup
    # as it requires interactive input of an MFA token. It often requires
    # manual activation post-creation or via custom scripts.
    # Leaving it commented out for general IaC deployments.
  }

  # Configure default server-side encryption for all objects in the bucket
  # Using AWS KMS for enhanced key management, auditing, and key rotation.
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm     = "aws:kms"
        kms_master_key_id = aws_kms_key.s3_encryption_key.arn # Reference the created KMS key
      }
    }
  }

  # Enforce 'BucketOwnerEnforced' object ownership.
  # This disables ACLs for the bucket and objects within it, simplifying access management
  # by ensuring that all objects are owned by the bucket owner. This is an AWS best practice.
  object_ownership = "BucketOwnerEnforced"

  # Standard tags for resource identification, management, and cost allocation
  tags = {
    Name        = "${var.bucket_name_prefix}-data"
    Environment = var.environment
    ManagedBy   = "Terraform"
    Purpose     = "ApplicationDataStorage"
  }
}

# Block all public access to the S3 bucket.
# This is a critical security measure to prevent accidental or malicious public exposure of data.
# It should be applied to almost all private S3 buckets.
resource "aws_s3_bucket_public_access_block" "production_bucket_public_access_block" {
  bucket = aws_s3_bucket.production_bucket.id

  # Prevents new public ACLs from being granted
  block_public_acls = true
  # Ignores existing public ACLs
  ignore_public_acls = true
  # Prevents new public bucket policies from being granted
  block_public_policy = true
  # Restricts access to only authorized AWS services or users
  restrict_public_buckets = true
}

# S3 Bucket Policy for the main application bucket.
# Explicitly defines *who* (e.g., specific IAM roles/users) can access the bucket
# and *what actions* (e.g., s3:GetObject, s3:PutObject) they can perform.
# This adds a critical layer of access control beyond IAM policies on principals.
resource "aws_s3_bucket_policy" "production_bucket_policy" {
  bucket = aws_s3_bucket.production_bucket.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Allow the AWS account root user or administrators with full S3 access
      # to perform any action on the bucket. This is a common baseline for account owners.
      {
        Sid       = "AllowBucketOwnerFullAccess"
        Effect    = "Allow"
        Principal = { AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.production_bucket.arn,
          "${aws_s3_bucket.production_bucket.arn}/*"
        ]
      },
      # Dynamically add a statement to allow a specific application IAM role to perform
      # specific S3 actions. This grants necessary permissions for the application
      # to interact with objects in the bucket, following the principle of least privilege.
      dynamic "statement" {
        for_each = var.application_iam_role_arn != "" ? [1] : []
        content {
          Sid       = "AllowApplicationRoleAccess"
          Effect    = "Allow"
          Principal = { AWS = var.application_iam_role_arn }
          Action    = [
            "s3:GetObject",
            "s3:PutObject",
            "s3:DeleteObject",
            "s3:ListBucket",
            "s3:GetBucketLocation", # Added s3:GetBucketLocation for full S3 client compatibility
            "s3:ListBucketMultipartUploads",
            "s3:ListMultipartUploadParts"
          ]
          Resource = [
            aws_s3_bucket.production_bucket.arn,
            "${aws_s3_bucket.production_bucket.arn}/*"
          ]
        }
      }
      # You may need to add more statements for other AWS services (e.g., Lambda, EC2)
      # or cross-account access if your application architecture requires it.
    ]
  })
}

# S3 Lifecycle Configuration for the main application bucket.
# Defines rules for object transitions to lower-cost storage classes (e.g., S3 Intelligent-Tiering, Glacier)
# after a certain period, or to expire old object versions to optimize costs and meet data retention policies.
resource "aws_s3_bucket_lifecycle_configuration" "production_bucket_lifecycle" {
  bucket = aws_s3_bucket.production_bucket.id

  rule {
    id     = "current-version-data-lifecycle"
    status = "Enabled"

    # Transition current version objects to S3 Intelligent-Tiering after 30 days.
    # Intelligent-Tiering automatically moves objects between storage tiers (Standard, IA, Archive Access)
    # based on access patterns, optimizing costs without performance impact.
    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }

    # Expire objects after 365 days if they are no longer needed, to optimize costs.
    # This rule was previously commented out and is now enabled.
    expiration {
      days = 365
    }

    # Clean up incomplete multipart uploads after 7 days to prevent lingering parts
    # from incurring storage costs.
    abort_incomplete_multipart_upload_days = 7
  }

  rule {
    id     = "previous-versions-data-lifecycle"
    status = "Enabled"

    # Expire previous (noncurrent) versions of objects after 90 days.
    # This helps manage storage costs while retaining a history for recovery for a defined period.
    noncurrent_version_expiration {
      noncurrent_days = 90
    }

    # Optional: Transition noncurrent versions to lower-cost storage before expiring them.
    # This can further optimize costs if previous versions are rarely accessed but need longer retention.
    # noncurrent_version_transition {
    #   noncurrent_days = 30
    #   storage_class   = "STANDARD_IA"
    # }
    # noncurrent_version_transition {
    #   noncurrent_days = 60
    #   storage_class   = "GLACIER"
    # }
  }
}

# -----------------------------------------------------------------------------
# S3 Logging Bucket
# -----------------------------------------------------------------------------
# Dedicated S3 bucket for storing access logs from other S3 buckets.
# It is best practice to deliver access logs to a separate, highly restricted
# S3 bucket, ideally in a dedicated logging account.
resource "aws_s3_bucket" "logging_bucket" {
  bucket = "${var.logging_bucket_name_prefix}-${random_id.logging_bucket_suffix.hex}"

  # Enable S3 Versioning for log integrity and recovery
  versioning {
    enabled = true
  }

  # Enforce 'BucketOwnerEnforced' object ownership for logs
  # This simplifies access management by disabling ACLs for log objects.
  object_ownership = "BucketOwnerEnforced"

  # Server-side encryption for logs (SSE-S3 is generally sufficient for logs themselves)
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  tags = {
    Name        = "${var.logging_bucket_name_prefix}-data"
    Environment = var.environment
    ManagedBy   = "Terraform"
    Purpose     = "S3AccessLogs"
  }
}

# Block all public access for the logging S3 bucket.
# This is a critical security measure for any S3 bucket, especially logging buckets,
# to prevent accidental or malicious public exposure of audit data.
resource "aws_s3_bucket_public_access_block" "logging_bucket_public_access_block" {
  bucket = aws_s3_bucket.logging_bucket.id

  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

# S3 Lifecycle Configuration for the logging bucket.
# Defines rules to manage object storage costs and retention for logs.
# Typically, logs can be transitioned to lower-cost storage classes and expired after a set period.
resource "aws_s3_bucket_lifecycle_configuration" "logging_bucket_lifecycle" {
  bucket = aws_s3_bucket.logging_bucket.id

  rule {
    id     = "expire-and-transition-logs"
    status = "Enabled"

    # Transition current version log objects to lower-cost storage after 30 days
    transition {
      days          = 30
      storage_class = "GLACIER_IR" # For infrequently accessed logs
    }

    # Expire log objects after 365 days. Adjust based on compliance requirements.
    expiration {
      days = 365
    }

    # Clean up incomplete multipart uploads after 7 days to prevent lingering parts incurring costs
    abort_incomplete_multipart_upload_days = 7
  }

  rule {
    id     = "expire-previous-versions-of-logs"
    status = "Enabled"

    # Expire previous versions of log objects after 90 days to save costs while retaining recent history
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

# S3 Bucket Policy for the logging bucket.
# This policy is crucial to allow the S3 service to deliver logs to this bucket.
# It restricts the `s3:PutObject` action to the S3 service principal and ensures
# logs originate from the source account and specific S3 bucket.
# This resource is now defined AFTER aws_s3_bucket.production_bucket to resolve forward reference.
resource "aws_s3_bucket_policy" "logging_bucket_policy" {
  bucket = aws_s3_bucket.logging_bucket.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ServerAccessLogsPolicy"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action = "s3:PutObject"
        Resource = [
          "${aws_s3_bucket.logging_bucket.arn}/*"
        ]
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id,
            "s3:x-amz-acl"      = "bucket-owner-full-control" # Required by S3 for cross-bucket log delivery
          }
          ArnLike = {
            # Ensure logs are only delivered from the specific production bucket
            "aws:SourceArn" = aws_s3_bucket.production_bucket.arn
          }
        }
      },
      {
        Sid    = "S3ServerAccessLogsGetBucketAcl"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action = "s3:GetBucketAcl"
        Resource = [
          aws_s3_bucket.logging_bucket.arn
        ]
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
          ArnLike = {
            "aws:SourceArn" = aws_s3_bucket.production_bucket.arn
          }
        }
      }
    ]
  })
}


# S3 Access Logging Configuration for the main application bucket.
# Enables S3 access logs, configuring them to be delivered to the separate,
# highly restricted logging bucket created earlier. This provides an essential
# audit trail of all access requests to your bucket for security monitoring and compliance.
resource "aws_s3_bucket_logging_v2" "production_bucket_logging" {
  bucket = aws_s3_bucket.production_bucket.id

  target_bucket = aws_s3_bucket.logging_bucket.id
  # Optional: Define a prefix for log objects within the target bucket for better organization.
  target_prefix = "logs/${var.bucket_name_prefix}-${random_id.bucket_suffix.hex}/"
}


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
# Output the globally unique name of the main application S3 bucket for easy reference.
output "s3_bucket_name" {
  description = "The globally unique name of the main application S3 bucket."
  value       = aws_s3_bucket.production_bucket.bucket
}

# Output the ARN (Amazon Resource Name) of the main application S3 bucket.
output "s3_bucket_arn" {
  description = "The ARN of the main application S3 bucket."
  value       = aws_s3_bucket.production_bucket.arn
}

# Output the globally unique name of the S3 logging bucket.
output "s3_logging_bucket_name" {
  description = "The globally unique name of the S3 logging bucket."
  value       = aws_s3_bucket.logging_bucket.bucket
}

# Output the ARN of the KMS key used for S3 encryption.
output "kms_key_arn" {
  description = "The ARN of the KMS key used for S3 encryption."
  value       = aws_kms_key.s3_encryption_key.arn
}

# Output the ARN of the KMS key alias for S3 encryption.
output "kms_key_alias_arn" {
  description = "The ARN of the KMS key alias used for S3 encryption."
  value       = aws_kms_alias.s3_encryption_alias.arn
}