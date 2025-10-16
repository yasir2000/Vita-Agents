import { useState } from 'react'
import Head from 'next/head'
import {
  AppBar,
  Box,
  Container,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  Alert,
} from '@mui/material'
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Psychology as AgentsIcon,
  Workflow as WorkflowIcon,
  Assessment as AnalyticsIcon,
  Settings as SettingsIcon,
  Health as HealthIcon,
} from '@mui/icons-material'
import { useQuery } from '@tanstack/react-query'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

// API functions
const fetchAgentStatus = async () => {
  const response = await fetch('/api/v1/agents')
  if (!response.ok) {
    throw new Error('Failed to fetch agent status')
  }
  return response.json()
}

const fetchHealthCheck = async () => {
  const response = await fetch('/health')
  if (!response.ok) {
    throw new Error('Failed to fetch health status')
  }
  return response.json()
}

export default function Home() {
  const [drawerOpen, setDrawerOpen] = useState(false)

  // Queries
  const { data: agents, isLoading: agentsLoading, error: agentsError } = useQuery({
    queryKey: ['agents'],
    queryFn: fetchAgentStatus,
    refetchInterval: 5000,
  })

  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealthCheck,
    refetchInterval: 10000,
  })

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, href: '/' },
    { text: 'Agents', icon: <AgentsIcon />, href: '/agents' },
    { text: 'Workflows', icon: <WorkflowIcon />, href: '/workflows' },
    { text: 'Analytics', icon: <AnalyticsIcon />, href: '/analytics' },
    { text: 'Settings', icon: <SettingsIcon />, href: '/settings' },
  ]

  // Sample data for charts
  const performanceData = [
    { time: '00:00', tasks: 12, success: 11 },
    { time: '04:00', tasks: 8, success: 8 },
    { time: '08:00', tasks: 25, success: 23 },
    { time: '12:00', tasks: 32, success: 30 },
    { time: '16:00', tasks: 28, success: 27 },
    { time: '20:00', tasks: 15, success: 14 },
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'success'
      case 'busy':
        return 'warning'
      case 'error':
        return 'error'
      default:
        return 'default'
    }
  }

  return (
    <>
      <Head>
        <title>Vita Agents - Healthcare AI Platform</title>
        <meta name="description" content="Multi-Agent AI Framework for Healthcare Interoperability" />
      </Head>

      <Box sx={{ display: 'flex' }}>
        {/* App Bar */}
        <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
          <Toolbar>
            <IconButton
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={() => setDrawerOpen(!drawerOpen)}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
            <HealthIcon sx={{ mr: 1 }} />
            <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
              Vita Agents
            </Typography>
            {health && (
              <Chip
                label={health.status}
                color={health.status === 'healthy' ? 'success' : 'error'}
                size="small"
              />
            )}
          </Toolbar>
        </AppBar>

        {/* Sidebar */}
        <Drawer
          variant="persistent"
          anchor="left"
          open={drawerOpen}
          sx={{
            width: 240,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: 240,
              boxSizing: 'border-box',
              top: 64,
            },
          }}
        >
          <List>
            {menuItems.map((item) => (
              <ListItem button key={item.text}>
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItem>
            ))}
          </List>
        </Drawer>

        {/* Main Content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            marginTop: 8,
            marginLeft: drawerOpen ? '240px' : 0,
            transition: 'margin-left 0.3s',
          }}
        >
          <Container maxWidth="xl">
            {/* Header */}
            <Typography variant="h4" gutterBottom>
              Dashboard
            </Typography>
            <Typography variant="subtitle1" color="text.secondary" gutterBottom>
              Multi-Agent AI Framework for Healthcare Interoperability
            </Typography>

            {/* Error Alert */}
            {agentsError && (
              <Alert severity="error" sx={{ mb: 2 }}>
                Failed to load agent data. Please check if the backend is running.
              </Alert>
            )}

            {/* Agent Status Cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Total Agents
                    </Typography>
                    <Typography variant="h3" color="primary">
                      {agents?.length || 0}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Active Agents
                    </Typography>
                    <Typography variant="h3" color="success.main">
                      {agents?.filter((a: any) => a.status === 'active').length || 0}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Total Tasks
                    </Typography>
                    <Typography variant="h3" color="info.main">
                      {agents?.reduce((sum: number, a: any) => sum + (a.metrics?.tasks_completed || 0), 0) || 0}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Success Rate
                    </Typography>
                    <Typography variant="h3" color="success.main">
                      {agents ? 
                        Math.round(
                          (agents.reduce((sum: number, a: any) => sum + (a.metrics?.tasks_completed || 0), 0) /
                          (agents.reduce((sum: number, a: any) => sum + (a.metrics?.tasks_completed || 0) + (a.metrics?.tasks_failed || 0), 0) || 1)) * 100
                        ) : 0}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Performance Chart */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Task Performance (24h)
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={performanceData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="tasks" stroke="#8884d8" name="Total Tasks" />
                        <Line type="monotone" dataKey="success" stroke="#82ca9d" name="Successful" />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>

              {/* Agent List */}
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Agent Status
                    </Typography>
                    {agentsLoading ? (
                      <Typography>Loading agents...</Typography>
                    ) : (
                      <List dense>
                        {agents?.map((agent: any) => (
                          <ListItem key={agent.agent_id}>
                            <ListItemIcon>
                              <AgentsIcon />
                            </ListItemIcon>
                            <ListItemText
                              primary={agent.name}
                              secondary={
                                <Chip
                                  label={agent.status}
                                  color={getStatusColor(agent.status)}
                                  size="small"
                                />
                              }
                            />
                          </ListItem>
                        ))}
                      </List>
                    )}
                  </CardContent>
                  <CardActions>
                    <Button size="small" color="primary">
                      View All
                    </Button>
                    <Button size="small" color="primary">
                      Manage
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            </Grid>

            {/* Quick Actions */}
            <Grid container spacing={3} sx={{ mt: 2 }}>
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Quick Actions
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                      <Button variant="contained" color="primary">
                        Validate FHIR Resource
                      </Button>
                      <Button variant="contained" color="secondary">
                        Convert HL7 to FHIR
                      </Button>
                      <Button variant="outlined" color="primary">
                        Run Workflow
                      </Button>
                      <Button variant="outlined" color="secondary">
                        EHR Integration
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Container>
        </Box>
      </Box>
    </>
  )
}