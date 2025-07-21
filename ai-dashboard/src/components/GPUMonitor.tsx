import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
} from '@mui/material';
import {
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';

interface GPUStats {
  gpu_available: boolean;
  device: string;
  gpu_memory_gb: number;
  gpu_memory_used_gb: number;
  gpu_memory_free_gb: number;
}

interface GPUMonitorProps {
  gpuStats: GPUStats | null;
  isBeastMode: boolean;
}

const GPUMonitor: React.FC<GPUMonitorProps> = ({ gpuStats, isBeastMode }) => {
  const memoryUsage = gpuStats ? (gpuStats.gpu_memory_used_gb / gpuStats.gpu_memory_gb) * 100 : 0;

  return (
    <Card sx={{ 
      height: '100%',
      background: isBeastMode 
        ? 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)' 
        : 'linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%)',
      border: `2px solid ${isBeastMode ? '#00d4ff' : '#ff6b35'}`,
    }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <MemoryIcon sx={{ mr: 1, color: isBeastMode ? '#00d4ff' : '#ff6b35' }} />
          <Typography variant="h6" sx={{ color: isBeastMode ? '#00d4ff' : '#ff6b35' }}>
            GPU Monitor
          </Typography>
          <Chip
            label={isBeastMode ? 'BEAST' : 'LIGHT'}
            size="small"
            color={isBeastMode ? 'primary' : 'secondary'}
            sx={{ ml: 'auto' }}
          />
        </Box>

        {gpuStats ? (
          <>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" sx={{ color: 'text.secondary', mb: 1 }}>
                Device: {gpuStats.device}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <CheckCircleIcon sx={{ color: '#00d4ff', mr: 1, fontSize: 16 }} />
                <Typography variant="body2" sx={{ color: '#00d4ff' }}>
                  GPU Available
                </Typography>
              </Box>
            </Box>

            <Box sx={{ mb: 3 }}>
              <Typography variant="body2" sx={{ color: 'text.secondary', mb: 1 }}>
                Memory Usage
              </Typography>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">
                  {gpuStats.gpu_memory_used_gb.toFixed(1)} GB
                </Typography>
                <Typography variant="body2">
                  {gpuStats.gpu_memory_gb.toFixed(1)} GB
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={memoryUsage}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  '& .MuiLinearProgress-bar': {
                    background: isBeastMode 
                      ? 'linear-gradient(90deg, #00d4ff 0%, #0099cc 100%)'
                      : 'linear-gradient(90deg, #ff6b35 0%, #cc5500 100%)',
                  },
                }}
              />
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                {memoryUsage.toFixed(1)}% used
              </Typography>
            </Box>

            <Box>
              <Typography variant="body2" sx={{ color: 'text.secondary', mb: 1 }}>
                Performance Mode
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <SpeedIcon sx={{ mr: 1, color: isBeastMode ? '#00d4ff' : '#ff6b35' }} />
                <Typography variant="body2">
                  {isBeastMode ? 'GPU Accelerated' : 'CPU Optimized'}
                </Typography>
              </Box>
            </Box>
          </>
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Loading GPU stats...
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default GPUMonitor; 