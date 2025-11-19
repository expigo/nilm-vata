// NILM VATA Dashboard - Frontend Application
// API Configuration
const API_BASE = window.location.origin;
const WS_BASE = `ws://${window.location.host}`;

// Global State
let currentTaskId = null;
let trainingPollInterval = null;
let websocket = null;
let mainsChart = null;
let appliancesChart = null;
let streamingData = {
    timestamps: [],
    mains: [],
    appliances: {}
};
const MAX_POINTS = 50; // Keep last 50 data points

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    console.log('NILM VATA Dashboard initializing...');

    await loadAlgorithms();
    await loadDatasets();
    await loadModels();
    initializeCharts();
    setupEventListeners();
});

// ============================================================================
// API Functions
// ============================================================================

async function apiGet(endpoint) {
    const response = await fetch(`${API_BASE}${endpoint}`);
    if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
    }
    return await response.json();
}

async function apiPost(endpoint, data) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    });
    if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
    }
    return await response.json();
}

// ============================================================================
// Load Data Functions
// ============================================================================

async function loadAlgorithms() {
    try {
        const data = await apiGet('/api/algorithms');
        const select = document.getElementById('algorithm-select');
        select.innerHTML = '<option value="">Select Algorithm...</option>';

        data.algorithms.forEach(algo => {
            const option = document.createElement('option');
            option.value = algo.id;
            option.textContent = `${algo.name} (${algo.type})`;
            option.dataset.info = JSON.stringify(algo);
            select.appendChild(option);
        });

        document.getElementById('stat-algorithms').textContent = data.algorithms.length;

        // Check for deep learning
        const hasDL = data.algorithms.some(a => a.type === 'deep_learning');
        document.getElementById('stat-dl').textContent = hasDL ? '✓ Available' : '✗ Not installed';

    } catch (error) {
        console.error('Failed to load algorithms:', error);
        showError('Failed to load algorithms');
    }
}

async function loadDatasets() {
    try {
        const data = await apiGet('/api/datasets');
        const select = document.getElementById('dataset-select');
        select.innerHTML = '<option value="">Select Dataset...</option>';

        let availableCount = 0;
        data.datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.id;
            option.textContent = `${dataset.name}${dataset.available ? '' : ' (not available)'}`;
            option.dataset.info = JSON.stringify(dataset.info);
            option.disabled = !dataset.available;
            select.appendChild(option);
            if (dataset.available) availableCount++;
        });

        document.getElementById('stat-datasets').textContent = `${availableCount}/${data.datasets.length}`;

    } catch (error) {
        console.error('Failed to load datasets:', error);
        showError('Failed to load datasets');
    }
}

async function loadModels() {
    try {
        const data = await apiGet('/api/models');
        const modelsList = document.getElementById('models-list');
        const modelSelect = document.getElementById('model-select');

        if (data.models.length === 0) {
            modelsList.innerHTML = '<p class="loading">No trained models yet. Train one above!</p>';
            modelSelect.innerHTML = '<option value="">No models available</option>';
        } else {
            modelsList.innerHTML = '';
            modelSelect.innerHTML = '<option value="">Select a model...</option>';

            data.models.forEach(model => {
                // Add to list
                const card = createModelCard(model);
                modelsList.appendChild(card);

                // Add to select
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = `${model.metadata.algorithm || 'Unknown'} - ${model.id}`;
                modelSelect.appendChild(option);
            });
        }

        document.getElementById('stat-models').textContent = data.models.length;

    } catch (error) {
        console.error('Failed to load models:', error);
        showError('Failed to load models');
    }
}

function createModelCard(model) {
    const card = document.createElement('div');
    card.className = 'model-card';
    card.onclick = () => selectModel(model.id);

    const metadata = model.metadata || {};
    const f1Score = metadata.avg_f1 ? (metadata.avg_f1 * 100).toFixed(1) : 'N/A';

    card.innerHTML = `
        <h4>${metadata.algorithm || 'Unknown Algorithm'}</h4>
        <p><strong>Model ID:</strong> ${model.id}</p>
        <p><strong>Dataset:</strong> ${metadata.dataset || 'Unknown'}</p>
        <p><strong>Avg F1 Score:</strong> ${f1Score}%</p>
    `;

    return card;
}

function selectModel(modelId) {
    document.getElementById('model-select').value = modelId;
    document.getElementById('start-stream-btn').disabled = false;
}

// ============================================================================
// Training Functions
// ============================================================================

async function startTraining() {
    const algorithm = document.getElementById('algorithm-select').value;
    const dataset = document.getElementById('dataset-select').value;
    const buildingId = parseInt(document.getElementById('building-id').value);
    const trainRatio = parseFloat(document.getElementById('train-ratio').value);

    if (!algorithm || !dataset) {
        alert('Please select both algorithm and dataset');
        return;
    }

    const trainBtn = document.getElementById('train-btn');
    trainBtn.disabled = true;
    trainBtn.textContent = '⏳ Training...';

    const statusBox = document.getElementById('training-status');
    statusBox.style.display = 'block';

    try {
        const response = await apiPost('/api/train', {
            algorithm,
            dataset,
            building_id: buildingId,
            train_ratio: trainRatio
        });

        currentTaskId = response.task_id;
        startTrainingPolling();

    } catch (error) {
        console.error('Failed to start training:', error);
        showError('Failed to start training: ' + error.message);
        trainBtn.disabled = false;
        trainBtn.textContent = '🚀 Start Training';
        statusBox.style.display = 'none';
    }
}

function startTrainingPolling() {
    trainingPollInterval = setInterval(async () => {
        try {
            const status = await apiGet(`/api/train/${currentTaskId}`);
            updateTrainingStatus(status);

            if (status.status === 'completed' || status.status === 'failed') {
                stopTrainingPolling();

                if (status.status === 'completed') {
                    showSuccess('Training completed successfully!');
                    await loadModels(); // Refresh models list
                } else {
                    showError('Training failed: ' + status.message);
                }
            }
        } catch (error) {
            console.error('Failed to poll training status:', error);
        }
    }, 1000); // Poll every second
}

function stopTrainingPolling() {
    if (trainingPollInterval) {
        clearInterval(trainingPollInterval);
        trainingPollInterval = null;
    }

    const trainBtn = document.getElementById('train-btn');
    trainBtn.disabled = false;
    trainBtn.textContent = '🚀 Start Training';
}

function updateTrainingStatus(status) {
    const progressFill = document.getElementById('progress-fill');
    const message = document.getElementById('training-message');

    progressFill.style.width = `${status.progress}%`;
    message.textContent = status.message;

    if (status.status === 'completed') {
        message.textContent = `✓ ${status.message} (F1: ${(status.avg_f1 * 100).toFixed(1)}%)`;
    }
}

// ============================================================================
// WebSocket & Streaming Functions
// ============================================================================

function startStreaming() {
    const modelId = document.getElementById('model-select').value;

    if (!modelId) {
        alert('Please select a model first');
        return;
    }

    const wsUrl = `${WS_BASE}/ws/disaggregate`;
    console.log('Connecting to WebSocket:', wsUrl);

    websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
        console.log('WebSocket connected');
        updateStreamStatus(true);

        // Send model configuration
        websocket.send(JSON.stringify({ model_id: modelId }));

        // Start simulated data stream
        startDataSimulation();
    };

    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.error) {
            showError('Streaming error: ' + data.error);
            stopStreaming();
            return;
        }

        if (data.status === 'ready') {
            console.log('Model loaded, ready for streaming');
            return;
        }

        // Update charts with disaggregated data
        if (data.appliances) {
            updateStreamingCharts(data);
        }
    };

    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        showError('WebSocket connection error');
    };

    websocket.onclose = () => {
        console.log('WebSocket disconnected');
        updateStreamStatus(false);
    };

    document.getElementById('start-stream-btn').disabled = true;
    document.getElementById('stop-stream-btn').disabled = false;
}

function stopStreaming() {
    if (websocket) {
        websocket.close();
        websocket = null;
    }

    if (simulationInterval) {
        clearInterval(simulationInterval);
        simulationInterval = null;
    }

    updateStreamStatus(false);
    document.getElementById('start-stream-btn').disabled = false;
    document.getElementById('stop-stream-btn').disabled = true;
}

function updateStreamStatus(connected) {
    const indicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');

    if (connected) {
        indicator.className = 'status-indicator connected';
        statusText.textContent = 'Connected - Streaming';
    } else {
        indicator.className = 'status-indicator disconnected';
        statusText.textContent = 'Disconnected';
    }
}

// Simulate streaming power data
let simulationInterval = null;

function startDataSimulation() {
    let time = 0;

    simulationInterval = setInterval(() => {
        if (!websocket || websocket.readyState !== WebSocket.OPEN) {
            return;
        }

        // Generate synthetic power value (500-3000W with realistic patterns)
        const baseLoad = 500;
        const variation = 2500 * Math.abs(Math.sin(time / 20));
        const noise = Math.random() * 200;
        const power = baseLoad + variation + noise;

        // Send to server
        websocket.send(JSON.stringify({ power }));

        time++;
    }, 500); // Send every 500ms
}

function updateStreamingCharts(data) {
    const timestamp = new Date(data.timestamp).toLocaleTimeString();

    // Update data buffers
    streamingData.timestamps.push(timestamp);
    streamingData.mains.push(data.mains);

    for (const [appName, power] of Object.entries(data.appliances)) {
        if (!streamingData.appliances[appName]) {
            streamingData.appliances[appName] = [];
        }
        streamingData.appliances[appName].push(power);
    }

    // Keep only last MAX_POINTS
    if (streamingData.timestamps.length > MAX_POINTS) {
        streamingData.timestamps.shift();
        streamingData.mains.shift();
        for (const appName in streamingData.appliances) {
            streamingData.appliances[appName].shift();
        }
    }

    // Update charts
    updateCharts();
    updatePowerStats(data);
}

// ============================================================================
// Chart Functions
// ============================================================================

function initializeCharts() {
    const mainsCtx = document.getElementById('mains-chart').getContext('2d');
    const appliancesCtx = document.getElementById('appliances-chart').getContext('2d');

    mainsChart = new Chart(mainsCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Total Power (W)',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                title: {
                    display: true,
                    text: 'Aggregate Power Consumption'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Power (W)'
                    }
                }
            }
        }
    });

    appliancesChart = new Chart(appliancesCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                title: {
                    display: true,
                    text: 'Disaggregated Appliances'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Power (W)'
                    }
                }
            }
        }
    });
}

const APPLIANCE_COLORS = {
    fridge: '#10b981',
    microwave: '#f59e0b',
    light: '#8b5cf6',
    dishwasher: '#ec4899',
    washing_machine: '#06b6d4',
    dryer: '#ef4444'
};

function updateCharts() {
    // Update mains chart
    mainsChart.data.labels = streamingData.timestamps;
    mainsChart.data.datasets[0].data = streamingData.mains;
    mainsChart.update('none'); // Update without animation for smoothness

    // Update appliances chart
    appliancesChart.data.labels = streamingData.timestamps;
    appliancesChart.data.datasets = [];

    const colorKeys = Object.keys(APPLIANCE_COLORS);
    let colorIndex = 0;

    for (const [appName, values] of Object.entries(streamingData.appliances)) {
        const color = APPLIANCE_COLORS[appName] || APPLIANCE_COLORS[colorKeys[colorIndex % colorKeys.length]];
        colorIndex++;

        appliancesChart.data.datasets.push({
            label: appName.replace('_', ' ').toUpperCase(),
            data: values,
            borderColor: color,
            backgroundColor: color + '20',
            borderWidth: 2,
            tension: 0.4,
            fill: true
        });
    }

    appliancesChart.update('none');
}

function updatePowerStats(data) {
    const statsContainer = document.getElementById('power-stats');
    statsContainer.innerHTML = '';

    // Total power
    const totalCard = document.createElement('div');
    totalCard.className = 'power-stat';
    totalCard.innerHTML = `
        <h4>Total Power</h4>
        <div class="value">${data.mains.toFixed(0)} W</div>
    `;
    statsContainer.appendChild(totalCard);

    // Individual appliances
    for (const [appName, power] of Object.entries(data.appliances)) {
        const card = document.createElement('div');
        card.className = 'power-stat';
        card.innerHTML = `
            <h4>${appName.replace('_', ' ').toUpperCase()}</h4>
            <div class="value">${power.toFixed(0)} W</div>
        `;
        statsContainer.appendChild(card);
    }
}

// ============================================================================
// Event Listeners
// ============================================================================

function setupEventListeners() {
    // Algorithm selection
    document.getElementById('algorithm-select').addEventListener('change', (e) => {
        const infoBox = document.getElementById('algorithm-info');
        if (e.target.value) {
            const info = JSON.parse(e.target.selectedOptions[0].dataset.info);
            infoBox.innerHTML = `
                <strong>Type:</strong> ${info.type}<br>
                <strong>Speed:</strong> ${info.speed}<br>
                <strong>Accuracy:</strong> ${info.accuracy}
            `;
            infoBox.classList.add('active');
        } else {
            infoBox.classList.remove('active');
        }
    });

    // Dataset selection
    document.getElementById('dataset-select').addEventListener('change', (e) => {
        const infoBox = document.getElementById('dataset-info');
        if (e.target.value) {
            const info = JSON.parse(e.target.selectedOptions[0].dataset.info);
            infoBox.innerHTML = Object.entries(info)
                .map(([key, value]) => `<strong>${key}:</strong> ${value}`)
                .join('<br>');
            infoBox.classList.add('active');
        } else {
            infoBox.classList.remove('active');
        }
    });

    // Train ratio slider
    document.getElementById('train-ratio').addEventListener('input', (e) => {
        document.getElementById('train-ratio-value').textContent =
            `${(e.target.value * 100).toFixed(0)}%`;
    });

    // Train button
    document.getElementById('train-btn').addEventListener('click', startTraining);

    // Stream control buttons
    document.getElementById('start-stream-btn').addEventListener('click', startStreaming);
    document.getElementById('stop-stream-btn').addEventListener('click', stopStreaming);
}

// ============================================================================
// Utility Functions
// ============================================================================

function showError(message) {
    console.error(message);
    // Could add a toast notification here
    alert('Error: ' + message);
}

function showSuccess(message) {
    console.log(message);
    // Could add a toast notification here
    alert('Success: ' + message);
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (websocket) {
        websocket.close();
    }
    if (trainingPollInterval) {
        clearInterval(trainingPollInterval);
    }
    if (simulationInterval) {
        clearInterval(simulationInterval);
    }
});
