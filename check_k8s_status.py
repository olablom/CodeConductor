#!/usr/bin/env python3
"""
🚀 CodeConductor Kubernetes Status Checker
Checks if Kubernetes is ready and provides deployment status
"""

import subprocess
import time
import sys
import os


def run_command(cmd, check=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        if check and result.returncode != 0:
            return None
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None


def check_kubernetes_status():
    """Check if Kubernetes is running"""
    print("🔍 Checking Kubernetes status...")

    # Check cluster info
    cluster_info = run_command("kubectl cluster-info", check=False)
    if cluster_info and "is running" in cluster_info:
        print("✅ Kubernetes is running!")
        return True
    else:
        print("⏳ Kubernetes is starting or not available")
        return False


def check_nodes():
    """Check Kubernetes nodes"""
    nodes = run_command("kubectl get nodes")
    if nodes:
        print("📊 Kubernetes Nodes:")
        print(nodes)
        return True
    return False


def check_namespaces():
    """Check existing namespaces"""
    namespaces = run_command("kubectl get namespaces")
    if namespaces:
        print("📦 Existing Namespaces:")
        print(namespaces)
        return True
    return False


def check_docker_images():
    """Check if our Docker images exist"""
    print("🐳 Checking Docker images...")

    images = [
        "codeconductor-gateway:latest",
        "codeconductor-agent:latest",
        "codeconductor-orchestrator:latest",
        "codeconductor-auth:latest",
        "codeconductor-data:latest",
    ]

    missing_images = []
    for image in images:
        result = run_command(f"docker images {image}", check=False)
        if not result or image not in result:
            missing_images.append(image)

    if missing_images:
        print(f"⚠️  Missing images: {', '.join(missing_images)}")
        print("💡 Run: ./k8s/prepare-demo.sh to build images")
        return False
    else:
        print("✅ All Docker images are ready!")
        return True


def main():
    """Main function"""
    print("🚀 CodeConductor Kubernetes Status Checker")
    print("=" * 50)

    # Check Kubernetes
    k8s_ready = check_kubernetes_status()

    if k8s_ready:
        print("\n📊 Kubernetes Details:")
        check_nodes()
        print()
        check_namespaces()
        print()
        check_docker_images()

        print("\n🎯 Next Steps:")
        print("1. Run: ./k8s/prepare-demo.sh")
        print("2. Run: cd k8s && ./deploy-demo.sh")
        print("3. Access: http://localhost:9000")

    else:
        print("\n⏳ Waiting for Kubernetes to start...")
        print("💡 Make sure Docker Desktop is running and Kubernetes is enabled")
        print("💡 This can take 2-3 minutes on first startup")

        # Wait and retry
        print("\n🔄 Retrying in 30 seconds...")
        time.sleep(30)

        if check_kubernetes_status():
            print("🎉 Kubernetes is now ready!")
            main()  # Recursive call to show full status
        else:
            print(
                "❌ Kubernetes still not ready. Please check Docker Desktop settings."
            )


if __name__ == "__main__":
    main()
