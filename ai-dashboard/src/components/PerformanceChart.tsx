import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';
import {
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';

interface AIDecision {
  id: string;
  timestamp: string;
  selected_arm: string;
  confidence: number;
  gpu_used: boolean;
  inference_time_ms: number;
  exploration: boolean;
}

interface PerformanceChartProps {
  aiDecisions: AIDecision[];
  isBeastMode: boolean;
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({ aiDecisions, isBeastMode }) => {
  // Transform data for chart
  const chartData = aiDecisions.map((decision, index) => ({
    time: index,
    confidence: decision.confidence * 100,
    inferenceTime: decision.inference_time_ms,
    arm: decision.selected_arm,
    exploration: decision.exploration ? 1 : 0,
  })).reverse();

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box
          sx={{
            backgroundColor: '#1a1a1a',
            border: '1px solid #00d4ff',
            borderRadius: 1,
            p: 2,
          }}
        >
          <Typography variant="body2" sx={{ color: '#00d4ff' }}>
            Decision {label}
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            Confidence: {payload[0]?.value?.toFixed(1)}%
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            Inference: {payload[1]?.value?.toFixed(2)}ms
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            Arm: {payload[2]?.payload?.arm}
          </Typography>
        </Box>
      );
    }
    return null;
  };

  return (
    <Card sx={{ 
      height: '100%',
      background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
      border: `2px solid ${isBeastMode ? '#00d4ff' : '#ff6b35'}`,
    }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <TrendingUpIcon sx={{ mr: 1, color: isBeastMode ? '#00d4ff' : '#ff6b35' }} />
          <Typography variant="h6" sx={{ color: isBeastMode ? '#00d4ff' : '#ff6b35' }}>
            AI Performance
          </Typography>
        </Box>

        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={isBeastMode ? "#00d4ff" : "#ff6b35"} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={isBeastMode ? "#00d4ff" : "#ff6b35"} stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="inferenceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ff6b35" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#ff6b35" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis 
                dataKey="time" 
                stroke="#666"
                tick={{ fill: '#666' }}
              />
              <YAxis 
                yAxisId="left"
                stroke="#00d4ff"
                tick={{ fill: '#00d4ff' }}
              />
              <YAxis 
                yAxisId="right" 
                orientation="right"
                stroke="#ff6b35"
                tick={{ fill: '#ff6b35' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="confidence"
                stroke={isBeastMode ? "#00d4ff" : "#ff6b35"}
                fillOpacity={1}
                fill="url(#confidenceGradient)"
                yAxisId="left"
                name="Confidence %"
              />
              <Line
                type="monotone"
                dataKey="inferenceTime"
                stroke="#ff6b35"
                strokeWidth={2}
                yAxisId="right"
                name="Inference Time (ms)"
                dot={{ fill: '#ff6b35', strokeWidth: 2, r: 4 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            height: 300,
            flexDirection: 'column',
          }}>
            <Typography variant="body2" sx={{ color: 'text.secondary', mb: 1 }}>
              Waiting for AI decisions...
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              Performance data will appear here
            </Typography>
          </Box>
        )}

        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
          <Box>
            <Typography variant="caption" sx={{ color: '#00d4ff' }}>
              Confidence
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" sx={{ color: '#ff6b35' }}>
              Inference Time
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default PerformanceChart; 