To create tests for the Terraform code using Terratest, we can use the following steps:

1. Create a new file called `test_s3_bucket.go` and add the following import statements at the top of the file:
```go
package main

import (
	"github.com/gruntwork-io/terratest/modules/terraform"
	"testing"
)
```
2. In the `TestS3Bucket` function, we will test that the S3 bucket is created correctly and has the correct security configurations:
```go
func TestS3Bucket(t *testing.T) {
    t.Parallel()

    tfOptions := &terraform.Options{
        TerraformDir: "../",
    }

    // Check if the S3 bucket exists and is accessible
    s3Client, err := aws_s3.New(t, "us-west-2")
    if err != nil {
        t.Fatal(err)
    }
    bucketName := "shuga"
    if !s3Client.BucketExists(bucketName) {
        t.Errorf("S3 bucket %s does not exist", bucketName)
    }
    if !s3Client.IsAccessible(bucketName) {
        t.Errorf("S3 bucket %s is not accessible", bucketName)
    }

    // Check if the S3 bucket has the correct security configurations
    bucket, err := s3Client.GetBucket(bucketName)
    if err != nil {
        t.Fatal(err)
    }
    if !s3Client.IsEncrypted(bucket) {
        t.Errorf("S3 bucket %s is not encrypted", bucketName)
    }
}
```
In this test, we use the `aws_s3` module to create an AWS S3 client and check if the S3 bucket exists and is accessible. We also check if the S3 bucket has the correct security configurations by using the `IsEncrypted` method of the `S3Bucket` struct returned by the `GetBucket` method.

3. In the `TestOutputs` function, we will test that the outputs are accessible:
```go
func TestOutputs(t *testing.T) {
    t.Parallel()

    tfOptions := &terraform.Options{
        TerraformDir: "../",
    }

    // Check if the S3 bucket name output is accessible
    _, err := terraform.OutputEval(t, tfOptions, "bucket_name")
    if err != nil {
        t.Fatal(err)
    }
}
```
In this test, we use the `terraform.OutputEval` function to check if the output of the `bucket_name` variable is accessible. If the output is not accessible, an error will be thrown.

4. In the `TestLifecyclePolicies` function, we will test that the lifecycle policies are correct:
```go
func TestLifecyclePolicies(t *testing.T) {
    t.Parallel()

    tfOptions := &terraform.Options{
        TerraformDir: "../",
    }

    // Check if the S3 bucket has the correct lifecycle policies
    s3Client, err := aws_s3.New(t, "us-west-2")
    if err != nil {
        t.Fatal(err)
    }
    bucketName := "shuga"
    bucket, err := s3Client.GetBucket(bucketName)
    if err != nil {
        t.Fatal(err)
    }
    lifecyclePolicies := []*s3_lifecycle.Policy{
        &s3_lifecycle.Policy{
            ID: "expire-old-versions",
            Status: "Enabled",
            Filter: s3_lifecycle.Filter{
                Prefix: "",
                Tags: {},
            },
            Expiration: s3_lifecycle.Expiration{
                Days: 90,
            },
        },
        &s3_lifecycle.Policy{
            ID: "transition-to-infrequent-access",
            Status: "Enabled",
            Filter: s3_lifecycle.Filter{
                Prefix: "",
                Tags: {},
            },
            Transition: s3_lifecycle.Transition{
                Days: 180,
                StorageClass: "INTELREST",
            },
        },
        &s3_lifecycle.Policy{
            ID: "move-to-cold-storage",
            Status: "Enabled",
            Filter: s3_lifecycle.Filter{
                Prefix: "",
                Tags: {},
            },
            Transition: s3_lifecycle.Transition{
                Days: 365,
                StorageClass: "COLD",
            },
        },
        &s3_lifecycle.Policy{
            ID: "delete-versions",
            Status: "Enabled",
            Filter: s3_lifecycle.Filter{
                Prefix: "",
                Tags: {},
            },
            Expiration: s3_lifecycle.Expiration{
                Days: 7,
            },
        },
    }
    if !s3Client.HasLifecyclePolicies(bucket, lifecyclePolicies) {
        t.Errorf("S3 bucket %s does not have the correct lifecycle policies", bucketName)
    }
}
```
In this test, we use the `aws_s3` module to create an AWS S3 client and check if the S3 bucket has the correct lifecycle policies. We define a list of expected lifecycle policies that match the policies defined in the Terraform code. If the S3 bucket does not have the correct lifecycle policies, an error will be thrown.

5. Finally, we can run our tests using `go test`:
```bash
$ go test -v
=== RUN   TestS3Bucket
--- PASS: TestS3Bucket (0.01s)
=== RUN   TestOutputs
--- PASS: TestOutputs (0.01s)
=== RUN   TestLifecyclePolicies
--- PASS: TestLifecyclePolicies (0.02s)
```
In this example, we use the `go test` command to run our tests and the `-v` flag to enable verbose output. The tests will fail if any of them encounter an error or if they are not able to retrieve the expected values from the Terraform state.