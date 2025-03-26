  // Initialize line chart (Bandwidth)
  const bandwidthCtx = document.getElementById('bandwidthChart').getContext('2d');
  const bandwidthData = {
      labels: [],  // Time labels
      datasets: [{
          label: 'Bandwidth (Kbps)',
          data: [],
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 2,
          fill: false
      }]
  };

  const pointsCtx = document.getElementById('pointsChart').getContext('2d');
  const pointsData = {
      labels: [],  // Time labels
      datasets: [{
          label: '# Points',
          data: [],
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 2,
          fill: false
      }]
  };

  const bandwidthConfig = {
      type: 'line',
      data: bandwidthData,
      options: {
          responsive: true,
          scales: {
              x: {
                  title: {
                      display: true,
                      text: 'Time (s)'
                  }
              },
              y: {
                  title: {
                      display: true,
                      text: 'Bandwidth (Kbps)'
                  },
                  beginAtZero: true
              }
          }
      }
  };

  const pointsConfig = {
    type: 'line',
    data: pointsData,
    options: {
        responsive: true,
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Time (s)'
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Number of Points'
                },
                beginAtZero: true
            }
        }
    }
  };

  const bandwidthChart = new Chart(bandwidthCtx, bandwidthConfig);
  const pointsChart = new Chart(pointsCtx, pointsConfig);

  // Initialize bar chart (Latencies)
  const enclatencyCtx = document.getElementById('encLatencyChart').getContext('2d');
  const declatencyCtx = document.getElementById('decLatencyChart').getContext('2d');
  const encLatencyData = {
      labels: ["Encoder Latencies"], // Single bar label
      datasets: [
          {
              label: 'E1: Analysis Transform',
              data: [10], // Value for Time 1
              backgroundColor: 'rgba(215, 48, 39, 0.8)',
              borderColor: 'rgb(0, 0, 0)',
              borderWidth: 1
          },
          {
              label: 'E2: Hyper Analysis Transform',
              data: [15], // Value for Time 2
              backgroundColor: 'rgba(244, 109, 67, 0.8)',
              borderColor: 'rgb(0, 0, 0)',
              borderWidth: 1
          },
          {
              label: 'E3: Factorized Bottleneck',
              data: [20], // Value for Time 3
              backgroundColor: 'rgba(253, 174, 97, 0.8)',
              borderColor: 'rgb(0, 0, 0)',
              borderWidth: 1
          },
          {
              label: 'E4: Hyper Synthesis Transform',
              data: [5], // Value for Time 4
              backgroundColor: 'rgba(255, 255, 191, 0.8)',
              borderColor: 'rgb(0, 0, 0)',
              borderWidth: 1
          },
          {
              label: 'E5: Gaussian Bottleneck',
              data: [10], // Value for Time 5
              backgroundColor: 'rgba(223, 243, 248, 0.8)',
              borderColor: 'rgb(0, 0, 0)',
              borderWidth: 1
          },
          {
              label: 'E6: G-PCC',
              data: [15], // Value for Time 6
              backgroundColor: 'rgba(115, 173, 210, 0.8)',
              borderColor: 'rgb(0, 0, 0)',
              borderWidth: 1
          },
          {
              label: 'E7: Bitstream Writing',
              data: [15], // Value for Time 7
              backgroundColor: 'rgba(69, 117, 180, 0.8)',
              borderColor: 'rgb(0, 0, 0)',
              borderWidth: 1
          }
      ]
  }; 
  
  const decLatencyData = {
      labels: ["Decoder Latencies"], // Single bar label
      datasets: [
          {
              label: 'D1: Bitstream Reading',
              data: [10], // Value for Time 1
              backgroundColor: 'rgba(70, 117, 180, 0.8)',
              borderColor: 'rgb(0, 0, 0)',
              borderWidth: 1
          },
          {
              label: 'D2: G-PCC',
              data: [15], // Value for Time 2
              backgroundColor: 'rgba(115, 173, 209, 0.8)',
              borderColor: 'rgb(0, 0, 0)',
              borderWidth: 1
          },
          {
              label: 'D3: Factorized Bottleneck',
              data: [20], // Value for Time 3
              backgroundColor: 'rgba(253, 174, 97, 0.8)',
              borderColor: 'rgb(0, 0, 0)',
              borderWidth: 1
          },
          {
              label: 'D4: Hyper Synthesis Transform',
              data: [5], // Value for Time 4
              backgroundColor: 'rgba(254, 223, 144, 0.8)',
              borderColor: 'rgb(0, 0, 0)',
              borderWidth: 1
          },
          {
              label: 'D5: Gaussian Bottleneck',
              data: [10], // Value for Time 5
              backgroundColor: 'rgba(255, 255, 191, 0.8)',
              borderColor: 'rgb(0, 0, 0)',
              borderWidth: 1
          },
          {
              label: 'D6: Synthesis Transform',
              data: [15], // Value for Time 6
              backgroundColor: 'rgba(215, 48, 38, 0.8)',
              borderColor: 'rgb(0, 0, 0)',
              borderWidth: 1
          }
      ]
  };

  const decLatencyConfig= {
      type: 'bar',
      data: decLatencyData,
      options: {
          responsive: true,
          plugins: {
              tooltip: {
                  enabled: true,
              },
              legend: {
                  position: 'top', // Position of the legend
                  labels: {
                    font: {
                        size: 10,
                    }
                  }
              }
          },
          scales: {
              x: {
                  stacked: true, // Enable stacking on x-axis
              },
              y: {
                  stacked: true, // Enable stacking on y-axis
                  title: {
                      display: true,
                      text: 'Latency (s)'
                  },
                  beginAtZero: true,
                  suggestedMin: 0,
                  suggestedMax: 1.0,
                  ticks: {
                      callback: function(value) {
                          return value + "s"
                      }
                  }
              }
          }
      }
      
  };
  const encLatencyConfig= {
      type: 'bar',
      data: encLatencyData,
      options: {
          responsive: true,
          plugins: {
              tooltip: {
                  enabled: true,
              },
              legend: {
                  position: 'top', // Position of the legend
                  labels: {
                    font: {
                        size: 10,
                    }
                  }
              }
          },
          scales: {
              x: {
                  stacked: true, // Enable stacking on x-axis
              },
              y: {
                  stacked: true, // Enable stacking on y-axis
                  title: {
                      display: true,
                      text: 'Latency (s)'
                  },
                  beginAtZero: true,
                  suggestedMin: 0,
                  suggestedMax: 1.0,
                  ticks: {
                      callback: function(value) {
                          return value + "s"
                      }
                  }
              }
          }
      }
  };
 

  const encLatencyChart = new Chart(enclatencyCtx, encLatencyConfig);
  const decLatencyChart = new Chart(declatencyCtx, decLatencyConfig);