import os

# Read the current GPU service main.py
gpu_service_path = "services/gpu_data_service/app/main.py"

if os.path.exists(gpu_service_path):
    with open(gpu_service_path, 'r') as f:
        content = f.read()
    
    # Check if metrics already exist
    if '/metrics' not in content:
        # Find the imports section and add our imports
        lines = content.split('\n')
        new_lines = []
        imports_added = False
        
        for line in lines:
            new_lines.append(line)
            # Add metrics imports after FastAPI import
            if 'from fastapi import' in line and not imports_added:
                new_lines.append('from fastapi.responses import Response')
                new_lines.append('from prometheus_client import Counter, Histogram, Gauge, generate_latest')
                new_lines.append('import time')
                new_lines.append('')
                new_lines.append('# Prometheus metrics')
                new_lines.append('REQUEST_COUNT = Counter("codeconductor_requests_total", "Total requests", ["service", "endpoint"])')
                new_lines.append('INFERENCE_DURATION = Histogram("codeconductor_inference_duration_seconds", "Inference duration", ["service", "algorithm"])')
                new_lines.append('GPU_MEMORY = Gauge("codeconductor_gpu_memory_bytes", "GPU memory usage", ["type"])')
                new_lines.append('BANDIT_SELECTIONS = Counter("codeconductor_bandit_selections_total", "Bandit arm selections", ["arm"])')
                new_lines.append('')
                imports_added = True
        
        # Add metrics endpoint before the end
        new_lines.append('')
        new_lines.append('@app.get("/metrics")')
        new_lines.append('async def metrics():')
        new_lines.append('    try:')
        new_lines.append('        import torch')
        new_lines.append('        if torch.cuda.is_available():')
        new_lines.append('            allocated = torch.cuda.memory_allocated()')
        new_lines.append('            reserved = torch.cuda.memory_reserved()')
        new_lines.append('            GPU_MEMORY.labels(type="allocated").set(allocated)')
        new_lines.append('            GPU_MEMORY.labels(type="reserved").set(reserved)')
        new_lines.append('    except:')
        new_lines.append('        pass')
        new_lines.append('    ')
        new_lines.append('    return Response(content=generate_latest(), media_type="text/plain")')
        
        # Write back
        with open(gpu_service_path, 'w') as f:
            f.write('\n'.join(new_lines))
        
        print("✅ Added metrics endpoint to GPU service")
    else:
        print("✅ Metrics endpoint already exists in GPU service")
else:
    print("❌ GPU service not found at", gpu_service_path)
