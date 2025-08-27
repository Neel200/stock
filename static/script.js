// --- Navigation ---
let customHistoryChart; // To hold the Chart.js instance for the custom history chart
let activeChartInstance = null;
let futureChartInstance = null; // To hold the Chart.js instance for future chart
const chartInstances = {}; // canvasId → Chart instance map
let customHistoryChartInstance = null;
const tabs = {
    'Home': 'tab-home',
    'Search': 'tab-search',
    'About': 'tab-about'
};
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        this.classList.add('active');

        Object.values(tabs).forEach(id => {
            const el = document.getElementById(id);
            el.classList.remove('active', 'animate__animated', 'animate__fadeIn');
        });

        const targetId = tabs[this.textContent.trim()];
        const target = document.getElementById(targetId);
        target.classList.add('active', 'animate__animated', 'animate__fadeIn');
    });
});
document.addEventListener('DOMContentLoaded', () => {
    const companyNameInput = document.getElementById('companyNameInput');
    const predictButton = document.getElementById('predictButton');
    const loadingDiv = document.getElementById('loading');
    const errorContainer = document.getElementById('errorContainer');
    const resultsDiv = document.getElementById('results'); // This is the main LSTM results div
    const companyNameDisplay = document.getElementById('companyNameDisplay');
    const metricsList = document.getElementById('metricsList');
    const fullPlot = document.getElementById('fullPlot');

    const historyTypeSelect = document.getElementById('historyTypeSelect');
    const historyValueSelect = document.getElementById('historyValueSelect');
    const showHistoryPlotButton = document.getElementById('showHistoryPlotButton');
    const customHistoryPlot = document.getElementById('customHistoryPlot');
    const downloadHistoryPlotButton = document.getElementById('downloadHistoryPlotButton');
    // ... (existing DOMContentLoaded code) ...

    // References to the custom history plot dropdowns and button
    // const customHistoryPlotImg = document.getElementById('customHistoryPlot'); // This is now a canvas, can be removed if not used for hiding/showing

    // Function to populate the duration dropdown based on type (Years/Months)
    // Function to fetch and display custom historical data chart
    // Renamed from getCustomHistoryPlot for better clarity as it now renders the chart
    async function renderCustomHistoryChart(symbol, startDate, endDate) { // Now accepts dates as parameters
        // The symbol, startDate, and endDate are now passed as arguments
        // No need to get them from input fields directly within this function

        if (!symbol || !startDate || !endDate) {
            // This check might be redundant if the calling logic ensures valid dates,
            // but it's good for robustness.
            showCustomAlert('Internal Error: Missing symbol, start date, or end date for chart rendering.'); // Changed from alert()
            return;
        }

        try {
            const response = await fetch('http://127.0.0.1:5000/get_custom_history_plot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol: symbol,
                    startDate: startDate,
                    endDate: endDate
                }),
            });

            const data = await response.json();

            if (data.error) {
                showCustomAlert('Error fetching custom historical data: ' + data.error); // Changed from alert()
                // Optionally hide canvas or show error on page
                // document.getElementById('customHistoryChart').classList.add('hidden');
                return;
            }

            // Make sure the canvas is visible
            document.getElementById('customHistoryChart').classList.remove('hidden');


            // Render chart using Chart.js
            const canvas = document.getElementById('customHistoryChart');
            const ctx = canvas.getContext('2d');

            // ✅ Destroy old Chart instance if it exists
            if (customHistoryChartInstance) {
                customHistoryChartInstance.destroy();
                customHistoryChartInstance = null;
            }

            // ✅ Create a new chart instance
            customHistoryChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.labels, // Dates
                    datasets: [{
                        label: 'Actual Prices (INR)',
                        data: data.actualData,
                        borderColor: 'red',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderWidth: 1,
                        pointRadius: 0, // No points for smoother line
                        fill: false
                    }, {
                        label: 'Predicted Prices (INR)',
                        data: data.predictedData,
                        borderColor: 'orange',
                        backgroundColor: 'rgba(255, 165, 0, 0.2)',
                        borderWidth: 1,
                        pointRadius: 0, // No points for smoother line
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'day'
                            },
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Price (INR)'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: `${symbol} Actual vs Predicted (${startDate} to ${endDate})`
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        },
                        zoom: { // Add zoom plugin configuration
                            pan: {
                                enabled: true,
                                mode: 'x'
                            },
                            zoom: {
                                wheel: {
                                    enabled: true,
                                },
                                pinch: {
                                    enabled: true
                                },
                                mode: 'xy',
                            },
                            pan: {
                                enabled: true,
                                mode: 'xy',
                            }
                        }
                    }
                }
            });

        } catch (error) {
            console.error('Error in renderCustomHistoryChart:', error);
            showCustomAlert('An error occurred while fetching custom historical data.'); // Changed from alert()
        }
    }

    function populateHistoryValueSelect() {
        const type = historyTypeSelect.value;
        historyValueSelect.innerHTML = ''; // Clear previous options

        if (type === 'years') {
            for (let i = 1; i <= 5; i++) { // For example, 1 to 5 years
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `${i} Year${i > 1 ? 's' : ''}`;
                historyValueSelect.appendChild(option);
            }
        } else if (type === 'months') {
            for (let i = 1; i <= 12; i++) { // For example, 1 to 12 months
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `${i} Month${i > 1 ? 's' : ''}`;
                historyValueSelect.appendChild(option);
            }
        }
    }

    // Initial population of the duration dropdown
    populateHistoryValueSelect();

    // Event listener for type change
    historyTypeSelect.addEventListener('change', populateHistoryValueSelect);

    // Event listener for the "Show Plot" button
    showHistoryPlotButton.addEventListener('click', async () => {
        const symbol = document.getElementById('companySymbolDisplay').textContent;
        const type = historyTypeSelect.value;
        const value = parseInt(historyValueSelect.value);

        if (!symbol) {
            showCustomAlert('Please perform a prediction first to get a company symbol.'); // Changed from alert()
            return;
        }

        if (type === 'custom') {
            const startDate = document.getElementById('historyStartDateLSTM').value;
            const endDate = document.getElementById('historyEndDateLSTM').value;

            if (!startDate || !endDate) {
                showCustomAlert('Please enter both start and end dates for the custom range.'); // Changed from alert()
                return;
            }

            await renderCustomHistoryChart(symbol, startDate, endDate);
        } else {
            const plotKey = `last_${value}_${type}`;
            renderTimeframeChart(plotKey); // <-- this is the key fix
        }
    });


    function renderTimeframeChart(plotKey) {
        const data = allPredictionPlots[`${plotKey}_data`];
        if (!data || !data.labels) {
            showCustomAlert(`No interactive chart data for ${plotKey.replace(/_/g, ' ')}`);
            return;
        }

        document.getElementById('customHistoryChart').classList.remove('hidden');

        renderChartJS(
            'customHistoryChart',
            data.labels,
            data.actualData,
            data.predictedData,
            `Actual vs Predicted (${plotKey.replace(/_/g, ' ')})`
        );
    }


    function renderChartJS(canvasId, labels, actualData, predictedData, title) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');

        // ✅ Destroy existing chart tied to this canvas
        if (chartInstances[canvasId]) {
            chartInstances[canvasId].destroy();
            delete chartInstances[canvasId];
        }

        // ✅ Create new chart instance and store it
        const newChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                        label: 'Actual Prices (INR)',
                        data: actualData,
                        borderColor: 'red',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: 'Predicted Prices (INR)',
                        data: predictedData,
                        borderColor: 'orange',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            displayFormats: {
                                day: 'MMM dd,yyyy' // This format will show "Jan 01, 2023"
                            }
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Price (INR)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: title
                    },
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'x'
                        },
                        zoom: {
                            wheel: {
                                enabled: true
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'x'
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    },
                    legend: {
                        position: 'top'
                    }
                },
                animation: {
                    duration: 500,
                    easing: 'easeOutQuart'
                }
            }

        });
        setTimeout(() => {
            const resetBtn = document.getElementById('resetZoomBtn');
            if (resetBtn) {
                resetBtn.onclick = () => stockChartInstance.resetZoom();
            }
        }, 100);
        chartInstances[canvasId] = newChart;
    }
    // ... (rest of your DOMContentLoaded code) ...

    // New DOM elements for live quote display
    const searchResultDiv = document.getElementById('searchResult'); // The div where the quote and chart will go

    let allPredictionPlots = {};
    let currentStockSymbol = ''; // To store the fetched symbol for chart rendering

    const forecastButtons = document.querySelectorAll('.forecast-btn');
    const customRangeInputDiv = document.getElementById('customRangeInput');
    const startDateInput = document.getElementById('startDate');
    const endDateInput = document.getElementById('endDate');
    const forecastCustomButton = document.getElementById('forecastCustomButton');
    // const futurePlot = document.getElementById('futurePlot'); // Removed as it's now a canvas
    const forecastTitle = document.getElementById('forecastTitle');
    const futurePredictionsTableBody = document.querySelector('#futurePredictionsTable tbody');

    let stockChartInstance = null; // To hold the Chart.js instance

    predictButton.addEventListener('click', async () => {
        const companyName = companyNameInput.value.trim();
        if (!companyName) {
            showCustomAlert('Please enter a company name.');
            return;
        }

        showLoading(true);
        hideError();
        searchResultDiv.innerHTML = ''; // Clear previous search results
        hideResults(); // Hide LSTM results initially

        try {
            // Step 1: Fetch live stock quote and overview
            const quoteResponse = await fetch('/get_stock_quote', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    companyName
                }),
            });

            const quoteData = await quoteResponse.json();

            if (!quoteResponse.ok) {
                showError(quoteData.error || 'Failed to fetch live stock data.');
                showLoading(false);
                return;
            }

            currentStockSymbol = quoteData.symbol; // Store symbol for chart
            const {
                quote,
                overview
            } = quoteData;

            // Render the live quote and chart
            displayLiveStockData(quote, overview);

            // Step 2: Trigger LSTM prediction
            const predictResponse = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    companyName
                }),
            });

            const predictionData = await predictResponse.json();

            if (predictResponse.ok) {
                // Display LSTM results within the predictionResult div
                displayLSTMResults(predictionData);
            } else {
                showError(predictionData.error || 'An unknown error occurred during prediction.');
            }
        } catch (error) {
            console.error('Error:', error);
            showError('Failed to connect to the server or process data. Please try again.');
        } finally {
            showLoading(false);
        }
    });

    forecastButtons.forEach(button => {
        button.addEventListener('click', async () => {
            const forecastType = button.dataset.type;
            const n = button.dataset.n;

            if (forecastType === 'custom') {
                customRangeInputDiv.classList.remove('hidden');
                return;
            } else {
                customRangeInputDiv.classList.add('hidden');
            }

            await requestForecast(forecastType, n);
        });
    });

    forecastCustomButton.addEventListener('click', async () => {
        const startDate = startDateInput.value;
        const endDate = endDateInput.value;

        if (!startDate || !endDate) {
            showCustomAlert('Please enter both start and end dates for custom forecast.');
            return;
        }
        await requestForecast('custom', null, startDate, endDate);
    });

    async function requestForecast(type, n = null, startDate = null, endDate = null) {
        showLoading(true);
        hideError();

        try {
            const response = await fetch('/forecast', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    type,
                    n,
                    startDate,
                    endDate
                }),
            });

            const data = await response.json();

            if (response.ok) {
                // Check if chart data is available and render Chart.js
                if (data.chart_data && data.chart_data.labels && data.chart_data.data) {
                    renderFutureChart(
                        data.chart_data.labels,
                        data.chart_data.data,
                        data.label // Use the label provided by the backend for the chart title
                    );
                    document.getElementById('futureChart').classList.remove('hidden'); // Ensure canvas is visible
                } else {
                    console.error("Forecast chart data not found in response.");
                    document.getElementById('futureChart').classList.add('hidden'); // Hide canvas if no data
                }

                if (data.label) {
                    forecastTitle.textContent = data.label + ":";
                } else {
                    forecastTitle.textContent = "Future Price Forecast:"; // Reset title
                }
                updateFuturePredictionsTable(data.table_data || []);
            } else {
                showError(data.error || 'An unknown error occurred during forecast.');
                document.getElementById('futureChart').classList.add('hidden'); // Hide canvas on error
                forecastTitle.textContent = "Future Price Forecast:"; // Reset title
                updateFuturePredictionsTable([]);
            }
        } catch (error) {
            console.error('Error:', error);
            showError('Failed to connect to the server for forecast.');
            document.getElementById('futureChart').classList.add('hidden'); // Hide canvas on network error
            forecastTitle.textContent = "Future Price Forecast:"; // Reset title
            updateFuturePredictionsTable([]);
        } finally {
            showLoading(false);
        }
    }

    // Function to display live stock data and chart
    function displayLiveStockData(quote, overview) {
        const change = quote["09. change"];
        const changePercent = quote["10. change percent"];
        const isPositive = parseFloat(change) >= 0;

        const changeBadge = `
            <span class="badge ${isPositive ? 'bg-success' : 'bg-danger'}">
                ${isPositive ? '▲' : '▼'} ${change} (${changePercent})
            </span>`;

        searchResultDiv.innerHTML = `
            <div class="result-card shadow p-4 animate__animated animate__fadeIn">
                <div class="card-body">
                    <h3 class="card-title mb-3">
                        <i class="bi bi-graph-up-arrow text-primary"></i> ${quote["01. symbol"]} (${overview["Name"] || 'N/A'})
                        ${changeBadge}
                    </h3>

                    ${generateQuoteDetails(quote)}

                    <div class="row justify-content-center mt-4">
                        <div class="col text-center">
                            ${generateTimeframeButtons()}
                        </div>
                    </div>

                    <div class="mt-4 chart-container">
                        <canvas id="stockChart" aria-label="Stock Price Chart"></canvas>
                        <button id="resetZoomButton" class="btn btn-outline-secondary btn-sm">Reset Zoom</button>
                    </div>

                    <div class="mt-4 text-start">
                        <p><strong>Sector:</strong> ${overview["Sector"] || "N/A"}</p>
                        <p><strong>Industry:</strong> ${overview["Industry"] || "N/A"}</p>
                        <p class="text-muted small">${overview["Description"] || "No description available."}</p>

                        <div id="predictionResult" class="mt-4"></div>
                    </div>
                </div>
            </div>`;

        // Attach event listeners for timeframe buttons after they are in the DOM
        document.querySelectorAll('.timeframe-btn').forEach(button => {
            button.addEventListener('click', async function () {
                const timeframe = this.dataset.timeframe;
                await renderChart(currentStockSymbol, timeframe);
                updateActiveButton(timeframe);
            });
        });

        // Render the initial daily chart
        renderChart(currentStockSymbol, 'daily');
        updateActiveButton('daily');
    }

    // New function to display LSTM prediction results separately
    function displayLSTMResults(data) {
        if (data.error) {
            // If there's an error in LSTM prediction, show it in the main error container or a specific div
            showError(`LSTM Prediction Error: ${data.error}`);
            return;
        }

        // Get the predictionResult div within the searchResultCard
        const predictionResultDiv = document.getElementById('predictionResult');
        if (!predictionResultDiv) {
            console.error("predictionResult div not found.");
            return;
        }

        // Build HTML for LSTM results
        let lstmHtml = `
            <h4 class="mb-3">LSTM Prediction & Forecast:</h4>
            <div class="metrics-section">
                <h3>Evaluation Metrics:</h3>
                <ul id="metricsListLSTM"></ul>
            </div>
            <div class="plots-section">
                <h3>Historical Performance:</h3>
                <div class="plot-container">
                    <h4>Full Prediction Plot:</h4>
                    <img id="fullPlotLSTM" src="" alt="Full Prediction Plot">
                </div>
                <div class="plot-container">
                    <h4>Custom History Plot (Actual vs Predicted):</h4>
                    <label for="historyTypeSelectLSTM">Select Type:</label>
                    <select id="historyTypeSelectLSTM">
                        <option value="years">Years</option>
                        <option value="months">Months</option>
                        <option value="custom">Custom Range</option> </select>
                    <div id="durationOptionsLSTM"> <label for="historyValueSelectLSTM">Select Duration:</label>
                        <select id="historyValueSelectLSTM"></select>
                    </div>
                    <div id="customHistoryRangeInputLSTM" class="hidden"> <label for="historyStartDateLSTM">Start Date:</label>
                        <input type="date" id="historyStartDateLSTM">
                        <label for="historyEndDateLSTM">End Date:</label>
                        <input type="date" id="historyEndDateLSTM">
                    </div>
                    <button id="showHistoryPlotButtonLSTM">Show Plot</button>
                    <button id="downloadHistoryPlotButtonLSTM" class="hidden">Download</button>
                    <canvas id="customHistoryChart" aria-label="Custom History Plot" class="hidden"></canvas>
                </div>
            </div>
            <div class="future-prediction-section">
                <h3>Future Price Forecast:</h3>
                <div class="forecast-options">
                    <button class="forecast-btn" data-type="months" data-n="1">Next 1 Month</button>
                    <button class="forecast-btn" data-type="months" data-n="3">Next 3 Months</button>
                    <button class="forecast-btn" data-type="months" data-n="6">Next 6 Months</button>
                    <button class="forecast-btn" data-type="year">Next 1 Year</button>
                    <button class="forecast-btn" data-type="custom">Custom Range</button>
                </div>
                <div id="customRangeInputLSTM" class="hidden">
                    <label for="startDateLSTM">Start Date:</label>
                    <input type="date" id="startDateLSTM">
                    <label for="endDateLSTM">End Date:</label>
                    <input type="date" id="endDateLSTM">
                    <button id="forecastCustomButtonLSTM">Forecast Custom Range</button>
                </div>
                <div class="plot-container">
                    <h4><span id="forecastTitleLSTM">Future Price Forecast</span>:</h4>
                    <!-- Updated to canvas -->
                    <canvas id="futureChartLSTM"></canvas> 
                </div>
                <div class="future-table-container">
                    <h4>Upcoming Predictions:</h4>
                    <table id="futurePredictionsTableLSTM">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Predicted Price</th>
                            </tr>
                        </thead>
                        <tbody>
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        predictionResultDiv.innerHTML = lstmHtml;

        // Populate LSTM Metrics
        // Populate LSTM Metrics
        const metricsListLSTM = document.getElementById('metricsListLSTM');
        metricsListLSTM.innerHTML = '';
        for (const [key, value] of Object.entries(data.metrics)) {
            // Skip 'Training Accuracy' and 'Testing Accuracy'
            if (key !== 'Training Accuracy' && key !== 'Testing Accuracy') {
                const li = document.createElement('li');
                // Ensure numerical values are formatted (e.g., to 4 decimal places)
                const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;
                li.textContent = `${key}: ${formattedValue}`;
                metricsListLSTM.appendChild(li);
            }
        }

        // Display LSTM Plots
        document.getElementById('fullPlotLSTM').src = `data:image/png;base64,${data.plots.full_plot}`;
        allPredictionPlots = data.plots; // Update global plots for history selection

        // Re-attach listeners for the LSTM-specific elements
        const historyTypeSelectLSTM = document.getElementById('historyTypeSelectLSTM');
        const historyValueSelectLSTM = document.getElementById('historyValueSelectLSTM');
        const showHistoryPlotButtonLSTM = document.getElementById('showHistoryPlotButtonLSTM');
        const customHistoryPlotLSTM = document.getElementById('customHistoryChart'); // This is now the canvas element
        const downloadHistoryPlotButtonLSTM = document.getElementById('downloadHistoryPlotButtonLSTM');
        const forecastButtonsLSTM = predictionResultDiv.querySelectorAll('.forecast-btn');
        const customRangeInputLSTM = document.getElementById('customRangeInputLSTM');
        const startDateLSTM = document.getElementById('startDateLSTM');
        const endDateLSTM = document.getElementById('endDateLSTM');
        const forecastCustomButtonLSTM = document.getElementById('forecastCustomButtonLSTM');
        // const futurePlotLSTM = document.getElementById('futurePlotLSTM'); // Removed as it's now a canvas
        const forecastTitleLSTM = document.getElementById('forecastTitleLSTM');
        const futurePredictionsTableBodyLSTM = document.querySelector('#futurePredictionsTableLSTM tbody');

        // Populate history options for LSTM
        populateHistoryOptionsLSTM(historyTypeSelectLSTM, historyValueSelectLSTM, allPredictionPlots);
        // Add or replace this block inside displayLSTMResults
        const durationOptionsLSTM = document.getElementById('durationOptionsLSTM'); // Get reference to the div
        const customHistoryRangeInputLSTM = document.getElementById('customHistoryRangeInputLSTM'); // Get reference to the div

        historyTypeSelectLSTM.addEventListener('change', () => {
            // This line calls populateHistoryOptionsLSTM which handles hiding/showing durationOptionsLSTM
            // when switching to/from 'custom' type.
            populateHistoryOptionsLSTM(historyTypeSelectLSTM, historyValueSelectLSTM, allPredictionPlots);

            if (historyTypeSelectLSTM.value === 'custom') {
                // If 'Custom Range' is selected:
                durationOptionsLSTM.classList.add('hidden'); // Hide the duration dropdown
                customHistoryRangeInputLSTM.classList.remove('hidden'); // Show the custom date inputs
            } else {
                // If 'Years' or 'Months' is selected:
                durationOptionsLSTM.classList.remove('hidden'); // Show the duration dropdown
                customHistoryRangeInputLSTM.classList.add('hidden'); // Hide the custom date inputs
            }
        });

        // script.js
        // ... inside displayLSTMResults function, inside showHistoryPlotButtonLSTM.addEventListener
        showHistoryPlotButtonLSTM.addEventListener('click', async () => {
            console.log('✅ Show Plot button clicked');
            const type = historyTypeSelectLSTM.value;
            const n = historyValueSelectLSTM.value;
            const plotKey = `last_${n}_${type}`;
            const startDate = historyStartDateLSTM.value;
            const endDate = historyEndDateLSTM.value;

            // Hide old image-based chart and download button
            // customHistoryPlotLSTM.classList.add('hidden'); // This might conflict if customHistoryPlotLSTM is the canvas now.
            downloadHistoryPlotButtonLSTM.classList.add('hidden'); // Hide download button if it was for image

            if (type === 'custom') {
                if (!startDate || !endDate) {
                    showCustomAlert('Please enter both start and end dates for custom history plot.');
                    return;
                }

                showLoading(true);
                try {
                    const response = await fetch('http://127.0.0.1:5000/get_custom_history_plot', { // Use full URL here as well
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            startDate,
                            endDate,
                            symbol: currentStockSymbol
                        }),
                    });

                    const data = await response.json();

                    if (response.ok && data.labels) {
                        // Hide base64 image (if it was an img tag)
                        // customHistoryPlotLSTM.classList.add('hidden'); // This is the canvas, don't hide it, just ensure it's visible.

                        // Render interactive Chart.js plot
                        renderChartJS(
                            'customHistoryChart', // Target the canvas
                            data.labels,
                            data.actualData,
                            data.predictedData,
                            `${currentStockSymbol} Actual vs Predicted (${startDate} to ${endDate})`
                        );
                        document.getElementById('customHistoryChart').classList.remove('hidden'); // Ensure canvas is visible

                    } else {
                        showCustomAlert(data.error || 'No data available for selected range.');
                    }
                } catch (error) {
                    console.error('Error fetching custom history plot:', error);
                    showCustomAlert('Failed to fetch custom history plot.');
                } finally {
                    showLoading(false);
                }

            } else {
                // Render interactive plot using precomputed plotKey
                renderTimeframeChart(plotKey);
            }
        });

        // Re-attach forecast listeners for LSTM
        forecastButtonsLSTM.forEach(button => {
            button.addEventListener('click', async () => {
                const forecastType = button.dataset.type;
                const n = button.dataset.n;

                if (forecastType === 'custom') {
                    customRangeInputLSTM.classList.remove('hidden');
                    return;
                } else {
                    customRangeInputLSTM.classList.add('hidden');
                }
                await requestForecastLSTM(forecastType, n, document.getElementById('futureChartLSTM'), forecastTitleLSTM, futurePredictionsTableBodyLSTM);
            });
        });

        forecastCustomButtonLSTM.addEventListener('click', async () => {
            const startDate = startDateLSTM.value;
            const endDate = endDateLSTM.value;
            if (!startDate || !endDate) {
                showCustomAlert('Please enter both start and end dates for custom forecast.');
                return;
            }
            await requestForecastLSTM('custom', null, document.getElementById('futureChartLSTM'), forecastTitleLSTM, futurePredictionsTableBodyLSTM, startDate, endDate);
        });
    }

    // Modified requestForecast for LSTM section
    async function requestForecastLSTM(type, n = null, plotCanvasElement, titleElement, tableBodyElement, startDate = null, endDate = null) {
        showLoading(true);
        hideError();

        try {
            const response = await fetch('/forecast', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    type,
                    n,
                    startDate,
                    endDate
                }),
            });
            const data = await response.json();

            if (response.ok) {
                if (data.chart_data && data.chart_data.labels && data.chart_data.data) {
                    renderFutureChart(
                        data.chart_data.labels,
                        data.chart_data.data,
                        data.label,
                        'futureChartLSTM' // Pass the specific canvas ID for LSTM forecast chart
                    );
                    plotCanvasElement.classList.remove('hidden'); // Ensure canvas is visible
                } else {
                    console.error("Forecast chart data not found in response.");
                    plotCanvasElement.classList.add('hidden'); // Hide canvas if no data
                }

                if (data.label) {
                    titleElement.textContent = data.label + ":";
                } else {
                    titleElement.textContent = "Future Price Forecast:";
                }
                updateFuturePredictionsTableLSTM(data.table_data || [], tableBodyElement);
            } else {
                showError(data.error || 'An unknown error occurred during forecast.');
                plotCanvasElement.classList.add('hidden'); // Hide canvas on error
                titleElement.textContent = "Future Price Forecast:";
                updateFuturePredictionsTableLSTM([], tableBodyElement);
            }
        } catch (error) {
            console.error('Error:', error);
            showError('Failed to connect to the server for forecast.');
            plotCanvasElement.classList.add('hidden'); // Hide canvas on network error
            titleElement.textContent = "Future Price Forecast:";
            updateFuturePredictionsTableLSTM([], tableBodyElement);
        } finally {
            showLoading(false);
        }
    }

    function updateFuturePredictionsTableLSTM(predictions, tableBodyElement) {
        tableBodyElement.innerHTML = '';
        if (predictions.length > 0) {
            predictions.forEach(p => {
                const row = tableBodyElement.insertRow();
                const dateCell = row.insertCell();
                const priceCell = row.insertCell();
                dateCell.textContent = p.date;
                priceCell.textContent = p.predicted_price;
            });
        } else {
            const row = tableBodyElement.insertRow();
            const cell = row.insertCell();
            cell.colSpan = 2;
            cell.textContent = "No future predictions available.";
        }
    }

    function populateHistoryOptionsLSTM(typeSelectElement, valueSelectElement, plotsData) {
        const type = typeSelectElement.value;
        const limit = type === 'years' ? 6 : 12;

        valueSelectElement.innerHTML = '';
        for (let i = 1; i <= limit; i++) {
            const key = `last_${i}_${type}_plot`;
            // Check for the presence of interactive chart data, not just the base64 image plot
            const isAvailable = plotsData[`last_${i}_${type}_data`] && plotsData[`last_${i}_${type}_data`].labels;

            const option = document.createElement('option');
            option.value = i;
            option.textContent = `${i} ${type}`;
            option.disabled = !isAvailable;
            valueSelectElement.appendChild(option);
        }
    }

    // Helper functions (unchanged, but added to this updated script)
    function showLoading(show) {
        if (show) {
            loadingDiv.classList.remove('hidden');
        } else {
            loadingDiv.classList.add('hidden');
        }
    }

    function showError(message) {
        errorContainer.textContent = message;
        errorContainer.classList.remove('hidden');
    }

    function hideError() {
        errorContainer.classList.add('hidden');
        errorContainer.textContent = '';
    }

    function hideResults() {
        resultsDiv.classList.add('hidden'); // The original LSTM results div
        // We'll manage the searchResultDiv visibility dynamically
    }

    // Custom Alert (replaces window.alert)
    function showCustomAlert(message) {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'custom-alert-message';
        alertDiv.innerHTML = `
            <p>${message}</p>
            <button class="custom-alert-close">OK</button>
        `;
        document.body.appendChild(alertDiv);

        alertDiv.querySelector('.custom-alert-close').addEventListener('click', () => {
            document.body.removeChild(alertDiv);
        });

        // Optional: Auto-hide after some time
        setTimeout(() => {
            if (document.body.contains(alertDiv)) {
                document.body.removeChild(alertDiv);
            }
        }, 5000);
    }

    // New helper functions for the quote/chart display

    function generateQuoteDetails(quote) {
        // Format numbers to INR currency
        const formatCurrency = (value) => {
            if (value === undefined || value === null || value === 'None' || value === '') {
                return 'N/A';
            }
            // Assume the value is already in INR with '₹' prefix from backend
            return value;
        };

        return `
            <div class="row text-center mt-3">
                <div class="col-md-4">
                    <p class="mb-0"><strong>Current Price:</strong></p>
                    <h4 class="text-primary">${formatCurrency(quote["05. price"])}</h4>
                </div>
                <div class="col-md-4">
                    <p class="mb-0"><strong>Open:</strong></p>
                    <p>${formatCurrency(quote["02. open"])}</p>
                </div>
                <div class="col-md-4">
                    <p class="mb-0"><strong>High:</strong></p>
                    <p>${formatCurrency(quote["03. high"])}</p>
                </div>
                <div class="col-md-4">
                    <p class="mb-0"><strong>Low:</strong></p>
                    <p>${formatCurrency(quote["04. low"])}</p>
                </div>
                <div class="col-md-4">
                    <p class="mb-0"><strong>Volume:</strong></p>
                    <p>${parseInt(quote["06. volume"]).toLocaleString() || 'N/A'}</p>
                </div>
                <div class="col-md-4">
                    <p class="mb-0"><strong>Latest Trading Day:</strong></p>
                    <p>${quote["07. latest trading day"] || 'N/A'}</p>
                </div>
            </div>
        `;
    }

    function generateTimeframeButtons() {
        return `
            <div id="timeframeButtons" class="btn-group" role="group" aria-label="Timeframe buttons">
                <button type="button" class="btn btn-outline-primary timeframe-btn active" data-timeframe="daily">Daily</button>
                <button type="button" class="btn btn-outline-primary timeframe-btn" data-timeframe="weekly">Weekly</button>
                <button type="button" class="btn btn-outline-primary timeframe-btn" data-timeframe="monthly">Monthly</button>
            </div>
        `;
    }

    async function renderChart(symbol, timeframe) {
        showLoading(true);
        try {
            const response = await fetch('/get_historical_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbol,
                    timeframe
                }),
            });
            const data = await response.json();

            if (!response.ok) {
                showError(data.error || 'Failed to fetch historical data for chart.');
                if (stockChartInstance) {
                    stockChartInstance.destroy();
                    stockChartInstance = null;
                }
                showLoading(false);
                return;
            }

            const ctx = document.getElementById('stockChart').getContext('2d');

            if (stockChartInstance) {
                stockChartInstance.destroy(); // Destroy previous chart instance
            }

            stockChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: `Close Price (${timeframe.charAt(0).toUpperCase() + timeframe.slice(1)})`,
                        data: data.data,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Price (INR)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                        },
                        tooltip: {
                            callbacks: {
                                label: function (context) {
                                    let label = context.dataset.label || '';
                                    if (label) label += ': ';
                                    if (context.parsed.y !== null) label += `₹${context.parsed.y.toFixed(2)}`;
                                    return label;
                                }
                            }
                        },
                        zoom: {
                            pan: {
                                enabled: true,
                                mode: 'x',
                            },
                            zoom: {
                                wheel: {
                                    enabled: true,
                                },
                                pinch: {
                                    enabled: true
                                },
                                mode: 'x',
                            },
                            limits: {
                                x: {
                                    minRange: 5
                                },
                            }
                        }
                    },
                    animation: {
                        duration: 500,
                        easing: 'easeInOutQuart',
                    },
                }

            });
        } catch (error) {
            console.error('Error rendering chart:', error);
            showError('Failed to load chart data.');
            if (stockChartInstance) {
                stockChartInstance.destroy();
                stockChartInstance = null;
            }
        } finally {
            showLoading(false);
        }
        setTimeout(() => {
            const resetBtn = document.getElementById('resetZoomBtn');
            if (resetBtn) {
                resetBtn.onclick = () => stockChartInstance.resetZoom();
            }
        }, 100); // Delay ensures the button is in DOM

    }
    // Add a new function for rendering the Future Price Forecast chart
    // To store the Chart.js instance for future chart
    // const resetFutureZoomBtn = document.getElementById('resetZoomBtn'); // This is the original one for futureChart
    const resetCustomHistoryZoomBtn = document.getElementById('resetCustomHistoryZoomBtn'); // The new one for customHistoryChart

    // Attach event listener for Future Chart's Reset Zoom
    // if (resetFutureZoomBtn) {
    //     resetFutureZoomBtn.addEventListener('click', () => {
    //         if (futureChartInstance) {
    //             futureChartInstance.resetZoom();
    //         }
    //     });
    // }

    // Attach event listener for Custom History Chart's Reset Zoom
    if (resetCustomHistoryZoomBtn) {
        resetCustomHistoryZoomBtn.addEventListener('click', () => {
            if (customHistoryChartInstance) {
                customHistoryChartInstance.resetZoom();
            }
        });
    }

    function renderFutureChart(labels, data, title, canvasId = 'futureChart') { // Added canvasId parameter
        const ctx = document.getElementById(canvasId).getContext('2d');

        // Dynamically select the correct futureChartInstance based on canvasId
        let currentFutureChartInstance;
        if (canvasId === 'futureChart') {
            currentFutureChartInstance = futureChartInstance;
        } else if (canvasId === 'futureChartLSTM') {
            currentFutureChartInstance = chartInstances['futureChartLSTM']; // Assuming you'd store it here
        }


        if (currentFutureChartInstance) {
            currentFutureChartInstance.destroy(); // Destroy previous chart instance if it exists
        }

        const newChartInstance = new Chart(ctx, { // Store the new instance
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Predicted Price (INR)',
                    data: data,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: title
                    },
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'x'
                        },
                        zoom: {
                            wheel: {
                                enabled: true
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'x'
                        },
                        pan: {
                            enabled: true,
                            mode: 'x', // Often only x-axis pan is preferred
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            displayFormats: {
                                day: 'MMM dd,yyyy' // Display day, month, and year
                            }
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Price (INR)'
                        }
                    }
                }
            }
        });

        // Update the correct futureChartInstance or store it in chartInstances map
        if (canvasId === 'futureChart') {
            futureChartInstance = newChartInstance;
        } else if (canvasId === 'futureChartLSTM') {
            chartInstances['futureChartLSTM'] = newChartInstance;
        }
        setTimeout(() => {
            const resetBtn = document.getElementById('resetZoomBtn');
            if (resetBtn) {
                resetBtn.onclick = () => stockChartInstance.resetZoom();
            }
        }, 100);
    }

    // Modify the forecast button click handler in script.js
    document.querySelectorAll('.forecast-btn').forEach(button => {
        button.addEventListener('click', async () => {
            const forecastType = button.dataset.type;
            const n = button.dataset.n;
            let startDate = null;
            let endDate = null;

            if (forecastType === 'custom') {
                customRangeInputDiv.classList.remove('hidden');
                startDate = startDateInput.value;
                endDate = endDateInput.value;
                if (!startDate || !endDate) {
                    showCustomAlert('Please enter both start and end dates for custom forecast.');
                    return;
                }
            } else {
                customRangeInputDiv.classList.add('hidden');
            }

            // Prepare forecast_options based on type and n
            const forecast_options = {
                type: forecastType,
                n: n,
                startDate: startDate,
                endDate: endDate
            };

            // Show loading spinner for forecast section if you have one
            // document.getElementById('forecastLoading').classList.remove('hidden');

            // Ensure the canvas is visible
            document.getElementById('futureChart').classList.remove('hidden');

            try {
                const response = await fetch('/forecast', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(forecast_options)
                });

                const data = await response.json();

                if (response.ok) {
                    if (data.chart_data && data.chart_data.labels && data.chart_data.data) {
                        renderFutureChart(
                            data.chart_data.labels,
                            data.chart_data.data,
                            data.label // Use the label provided by the backend for the chart title
                        );
                    } else {
                        console.error("Forecast chart data not found in response.");
                        // Optionally hide the canvas if no data
                        document.getElementById('futureChart').classList.add('hidden');
                    }

                    if (data.label) {
                        document.getElementById('forecastTitle').textContent = data.label + ":";
                    }

                    // Update the table data (existing logic)
                    const tableBody = document.querySelector('#futurePredictionsTable tbody');
                    tableBody.innerHTML = '';
                    if (data.table_data && data.table_data.length > 0) {
                        data.table_data.forEach(item => {
                            const row = tableBody.insertRow();
                            row.insertCell().textContent = item.date;
                            row.insertCell().textContent = item.predicted_price;
                        });
                    } else {
                        const row = tableBody.insertRow();
                        const cell = row.insertCell();
                        cell.colSpan = 2;
                        cell.textContent = "No future predictions available for this range.";
                    }

                } else {
                    console.error('Forecast error:', data.error);
                    // Display error message to user
                    showError(data.error || 'An unknown error occurred during forecast.');
                    document.getElementById('futureChart').classList.add('hidden'); // Hide canvas on error
                }
            } catch (error) {
                console.error('Network or parsing error:', error);
                // Display error message to user
                showError('Failed to connect to the server for forecast.');
                document.getElementById('futureChart').classList.add('hidden'); // Hide canvas on network error
            } finally {
                // Hide loading spinner
                // document.getElementById('forecastLoading').classList.add('hidden');
            }
        });
    });

    // IMPORTANT: You should also ensure that the initial prediction flow
    // (when the main 'Predict Stock Price' button is clicked) correctly
    // handles showing/hiding the initial state of the futureChart.
    // For example, when results are first shown, ensure futureChart is hidden
    // or has its initial state handled correctly.

    function updateActiveButton(timeframe) {
        document.querySelectorAll('.timeframe-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.timeframe === timeframe) {
                btn.classList.add('active');
            }
        });
    }

    // Add CSS for the custom alert (replace `alert()`)
    const style = document.createElement('style');
    style.innerHTML = `
        .custom-alert-message {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: white;
            border: 1px solid #ccc;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            z-index: 1000;
            text-align: center;
            border-radius: 8px;
            max-width: 300px;
        }
        .custom-alert-message p {
            margin-bottom: 15px;
            font-size: 16px;
        }
        .custom-alert-close {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
        }
        .custom-alert-close:hover {
            background-color: #0056b3;
        }
    `;
    document.head.appendChild(style);

});