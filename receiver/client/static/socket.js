// Connect to Socket.IO once
const socket = io();

// Update charts when new data arrives
socket.on('update_data', (data) => {
    // Update bandwidth line chart
    const now = new Date();
    const formattedTime = now.toLocaleTimeString('en-US', { hour12: false }); // Format as HH:mm:ss

    // Bandwidth Chart
    bandwidthData.labels.push(formattedTime);
    bandwidthData.datasets[0].data.push(data.bandwidth);

    if (bandwidthData.labels.length > 20) {
        bandwidthData.labels.shift(); // Remove the oldest label
    }
    if (bandwidthData.datasets[0].data.length > 20) {
        bandwidthData.datasets[0].data = bandwidthData.datasets[0].data.slice(-20); 
    }

    bandwidthChart.update({easing: 'easeIntCubic'});

    // Points Chart
    pointsData.labels.push(formattedTime);
    pointsData.datasets[0].data.push(data.points);

    if (pointsData.labels.length > 20) {
        pointsData.labels.shift(); // Remove the oldest label
    }
    if (pointsData.datasets[0].data.length > 20) {
        pointsData.datasets[0].data = pointsData.datasets[0].data.slice(-20); 
    }

    pointsChart.update({easing: 'easeIntCubic'});


    // Update latency bar chart
    const { e1, e2, e3, e4, e5, e6, e7, d1, d2, d3, d4, d5, d6} = data.latencies;
    encLatencyData.datasets[0].data = [e1];
    encLatencyData.datasets[1].data = [e2];
    encLatencyData.datasets[2].data = [e3];
    encLatencyData.datasets[3].data = [e4];
    encLatencyData.datasets[4].data = [e5];
    encLatencyData.datasets[5].data = [e6];
    encLatencyData.datasets[6].data = [e7];
    decLatencyData.datasets[0].data = [d1];
    decLatencyData.datasets[1].data = [d2];
    decLatencyData.datasets[2].data = [d3];
    decLatencyData.datasets[3].data = [d4];
    decLatencyData.datasets[4].data = [d5];
    decLatencyData.datasets[5].data = [d6];

    encLatencyChart.update();  
    decLatencyChart.update();  
});