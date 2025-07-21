import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Dashboard as DashboardIcon,
} from '@mui/icons-material';

interface SystemStatusProps {
  status: {
    gpuService: boolean;
    prometheus: boolean;
    grafana: boolean;
  };
}

const SystemStatus: React.FC<SystemStatusProps> = ({ status }) => {
  const services = [
    {
      name: 'GPU Service',
      status: status.gpuService,
      icon: <MemoryIcon />,
      color: 'primary' as const,
      description: 'RTX 5090 AI Engine',
    },
    {
      name: 'Prometheus',
      status: status.prometheus,
      icon: <SpeedIcon />,
      color: 'secondary' as const,
      description: 'Metrics Collection',
    },
    {
      name: 'Grafana',
      status: status.grafana,
      icon: <DashboardIcon />,
      color: 'success' as const,
      description: 'Dashboard Platform',
    },
  ];

  return (
    <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff', mb: 2 }}>
          System Status
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          {services.map((service) => (
            <Box
              key={service.name}
              sx={{
                flex: '1 1 300px',
                minWidth: 0,
                display: 'flex',
                alignItems: 'center',
                p: 2,
                borderRadius: 1,
                border: `1px solid ${service.status ? '#00d4ff' : '#ff6b35'}`,
                background: service.status ? 'rgba(0, 212, 255, 0.1)' : 'rgba(255, 107, 53, 0.1)',
              }}
            >
              <Box sx={{ mr: 2 }}>
                {service.status ? (
                  <CheckCircleIcon sx={{ color: '#00d4ff' }} />
                ) : (
                  <ErrorIcon sx={{ color: '#ff6b35' }} />
                )}
              </Box>
              <Box sx={{ flexGrow: 1 }}>
                <Typography variant="subtitle2" sx={{ color: service.status ? '#00d4ff' : '#ff6b35' }}>
                  {service.name}
                </Typography>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  {service.description}
                </Typography>
              </Box>
              <Chip
                label={service.status ? 'ONLINE' : 'OFFLINE'}
                size="small"
                color={service.status ? 'success' : 'error'}
                sx={{ ml: 1 }}
              />
            </Box>
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

export default SystemStatus; 