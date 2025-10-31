terraform {

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "3.60"
    }
  }

  provider "aws" {
    region = var.region
  }
}

resource "aws_s3_bucket" "shuga-bucket" {
  bucket        = var.bucket_name
  acl           = "private"
  force_destroy = true
  versioning {
    enabled = var.versioning
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "aws:kms"
      }
    }
  }
}
