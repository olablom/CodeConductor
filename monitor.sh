#!/bin/bash
# CodeConductor CI/CD Monitor (Shell Version)
# Runs every 5 minutes until everything is perfect

echo "🎼 CodeConductor CI/CD Monitor Started!"
echo "📊 Monitoring every 5 minutes until everything is perfect..."
echo "⏹️ Press Ctrl+C to stop"
echo "🔗 GitHub Actions: https://github.com/olablom/CodeConductor/actions"
echo ""

check_count=0

while true; do
    check_count=$((check_count + 1))
    current_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "=================================================================================="
    echo "🎼 CodeConductor CI/CD Monitor - $current_time"
    echo "=================================================================================="
    
    echo ""
    echo "🐳 Checking local services..."
    python check_ci.py
    
    echo ""
    echo "📊 Checking GitHub Actions..."
    python check_github_actions.py | head -20
    
    echo ""
    echo "🎯 Status Check #$check_count"
    echo "⏰ Next check in 5 minutes... (Press Ctrl+C to stop)"
    echo "💡 Manual checks:"
    echo "   • python check_ci.py"
    echo "   • python check_github_actions.py"
    echo "   • https://github.com/olablom/CodeConductor/actions"
    echo ""
    
    # Wait 5 minutes
    sleep 300
done 