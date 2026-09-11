document.addEventListener('DOMContentLoaded', () => {
    // Global Elements
    const statFiles = document.getElementById('stat-files');
    const statNotes = document.getElementById('stat-notes');
    const statVocab = document.getElementById('stat-vocab');
    const datasetBadge = document.getElementById('dataset-status-badge');
    const modelBadge = document.getElementById('model-status-badge');
    const alertContainer = document.getElementById('alert-container');

    const btnPreprocess = document.getElementById('btn-preprocess');
    const btnTrain = document.getElementById('btn-train');
    const btnGenerate = document.getElementById('btn-generate');

    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');

    const genNotesInput = document.getElementById('gen-notes');
    const genTempInput = document.getElementById('gen-temp');
    const valNotes = document.getElementById('val-notes');
    const valTemp = document.getElementById('val-temp');

    const trainingProgressContainer = document.getElementById('training-progress-container');
    const trainProgressBar = document.getElementById('train-progress-bar');
    const trainStatusText = document.getElementById('train-status-text');
    const trainEpochText = document.getElementById('train-epoch-text');

    const resultSection = document.getElementById('result-section');
    const resFilename = document.getElementById('res-filename');
    const btnDownload = document.getElementById('btn-download');
    const btnPlay = document.getElementById('btn-play');
    const btnStop = document.getElementById('btn-stop');
    const playerStatus = document.getElementById('player-status');

    // Chart Instances
    let lossChart = null;
    let accuracyChart = null;
    let pitchRollChart = null;

    // Tone.js State
    let synth = null;
    let currentTonePart = null;
    let isPlaying = false;

    // Range input listeners
    genNotesInput.addEventListener('input', (e) => valNotes.textContent = e.target.value);
    genTempInput.addEventListener('input', (e) => valTemp.textContent = parseFloat(e.target.value).toFixed(1));

    // Show alert notice
    function showAlert(message, type = 'info') {
        const wrapper = document.createElement('div');
        wrapper.className = `alert alert-${type} alert-dismissible fade show glass-card mb-4 text-white`;
        wrapper.setAttribute('role', 'alert');
        wrapper.innerHTML = `
            <div>${message}</div>
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        alertContainer.replaceChildren(wrapper);
    }

    // Initialize Charts
    function initCharts() {
        const ctxLoss = document.getElementById('lossChart').getContext('2d');
        lossChart = new Chart(ctxLoss, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Train Loss', data: [], borderColor: '#00f2fe', backgroundColor: 'rgba(0, 242, 254, 0.1)', fill: true, tension: 0.3 },
                    { label: 'Val Loss', data: [], borderColor: '#ff0844', backgroundColor: 'rgba(255, 8, 68, 0.1)', fill: true, tension: 0.3 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
                },
                plugins: { legend: { labels: { color: '#f8fafc' } } }
            }
        });

        const ctxAcc = document.getElementById('accuracyChart').getContext('2d');
        accuracyChart = new Chart(ctxAcc, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Train Accuracy', data: [], borderColor: '#7f53ac', backgroundColor: 'rgba(127, 83, 172, 0.1)', fill: true, tension: 0.3 },
                    { label: 'Val Accuracy', data: [], borderColor: '#22c55e', backgroundColor: 'rgba(34, 197, 94, 0.1)', fill: true, tension: 0.3 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
                },
                plugins: { legend: { labels: { color: '#f8fafc' } } }
            }
        });
    }

    function initPitchRollChart(notes) {
        const ctxRoll = document.getElementById('pitchRollChart').getContext('2d');
        if (pitchRollChart) pitchRollChart.destroy();

        // Convert note/chord strings into numeric MIDI pitch representation for charting
        const dataPoints = notes.map((item, idx) => {
            if (item.includes('.')) {
                // Return pitch average for chord
                const pitches = item.split('.').map(p => Tone.Frequency(p).toMidi() || 60);
                return { x: idx + 1, y: pitches[0] };
            } else {
                const pitch = Tone.Frequency(item).toMidi() || 60;
                return { x: idx + 1, y: pitch };
            }
        });

        pitchRollChart = new Chart(ctxRoll, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Generated Notes (Pitch vs Time)',
                    data: dataPoints,
                    backgroundColor: '#00f2fe',
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Note Sequence Index', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                    y: { title: { display: true, text: 'MIDI Pitch Number', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
                },
                plugins: { legend: { labels: { color: '#f8fafc' } } }
            }
        });
    }

    // Load initial dataset info
    async function loadDatasetInfo() {
        try {
            const res = await fetch('/api/dataset-info');
            const data = await res.json();

            if (data.success) {
                statFiles.textContent = data.num_files;
                if (data.is_preprocessed && data.processed_info) {
                    statNotes.textContent = data.processed_info.num_samples;
                    statVocab.textContent = data.processed_info.vocab_size;
                    datasetBadge.textContent = 'Preprocessed';
                    datasetBadge.className = 'badge bg-success-glow';
                } else {
                    datasetBadge.textContent = data.num_files > 0 ? 'Raw Loaded' : 'No Files';
                    datasetBadge.className = 'badge bg-secondary-glow';
                }

                if (data.model_exists) {
                    modelBadge.textContent = 'Trained Model Ready';
                    modelBadge.className = 'badge bg-success-glow';
                } else {
                    modelBadge.textContent = 'Untrained';
                    modelBadge.className = 'badge bg-secondary-glow';
                }
            }
        } catch (e) {
            console.error('Error fetching dataset info:', e);
        }
    }

    // Handle Upload
    fileInput.addEventListener('change', async (e) => {
        if (!e.target.files.length) return;
        const formData = new FormData();
        for (let file of e.target.files) {
            formData.append('file', file);
        }

        showAlert('Uploading MIDI file(s)...', 'info');
        try {
            const res = await fetch('/api/upload', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                showAlert(data.message, 'success');
                loadDatasetInfo();
            } else {
                showAlert(data.error, 'danger');
            }
        } catch (err) {
            showAlert('File upload failed: ' + err, 'danger');
        }
    });

    // Drag & Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        fileInput.dispatchEvent(new Event('change'));
    });

    // Handle Preprocess
    btnPreprocess.addEventListener('click', async () => {
        btnPreprocess.disabled = true;
        btnPreprocess.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i>Processing MIDI Files...';

        const seqLen = document.getElementById('cfg-seq-len').value;
        try {
            const res = await fetch('/api/preprocess', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sequence_length: parseInt(seqLen) })
            });
            const data = await res.json();
            if (data.success) {
                showAlert(`Preprocessing complete! ${data.num_samples} note sequences created with vocab size ${data.vocab_size}.`, 'success');
                loadDatasetInfo();
            } else {
                showAlert(data.error, 'danger');
            }
        } catch (err) {
            showAlert('Preprocessing error: ' + err, 'danger');
        } finally {
            btnPreprocess.disabled = false;
            btnPreprocess.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles me-2"></i>Preprocess Dataset';
        }
    });

    // Handle Train
    btnTrain.addEventListener('click', async () => {
        const payload = {
            sequence_length: parseInt(document.getElementById('cfg-seq-len').value),
            lstm_units: parseInt(document.getElementById('cfg-lstm-units').value),
            epochs: parseInt(document.getElementById('cfg-epochs').value),
            batch_size: parseInt(document.getElementById('cfg-batch-size').value)
        };

        btnTrain.disabled = true;
        btnTrain.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i>Starting Training...';
        trainingProgressContainer.classList.remove('d-none');

        try {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) {
                showAlert(data.message, 'info');
                pollTrainStatus();
            } else {
                showAlert(data.error, 'danger');
                btnTrain.disabled = false;
                btnTrain.innerHTML = '<i class="fa-solid fa-play me-2"></i>Start Training';
            }
        } catch (err) {
            showAlert('Training request failed: ' + err, 'danger');
            btnTrain.disabled = false;
            btnTrain.innerHTML = '<i class="fa-solid fa-play me-2"></i>Start Training';
        }
    });

    // Poll Training Progress
    let trainPollInterval = null;
    function pollTrainStatus() {
        if (trainPollInterval) clearInterval(trainPollInterval);

        trainPollInterval = setInterval(async () => {
            try {
                const res = await fetch('/api/train-status');
                const data = await res.json();

                if (data.success) {
                    const st = data.status;
                    const saved = data.saved_history || {};

                    const currEpoch = st.current_epoch || saved.current_epoch || 0;
                    const totalEpochs = st.total_epochs || saved.total_epochs || 1;
                    const pct = Math.round((currEpoch / totalEpochs) * 100);

                    trainProgressBar.style.width = `${pct}%`;
                    trainEpochText.textContent = `Epoch ${currEpoch}/${totalEpochs}`;
                    trainStatusText.textContent = st.message || 'Training...';

                    // Update Charts
                    const losses = st.loss.length ? st.loss : (saved.loss || []);
                    const valLosses = st.val_loss.length ? st.val_loss : (saved.val_loss || []);
                    const accs = st.accuracy.length ? st.accuracy : (saved.accuracy || []);
                    const valAccs = st.val_accuracy.length ? st.val_accuracy : (saved.val_accuracy || []);

                    const labels = Array.from({ length: losses.length }, (_, i) => `Epoch ${i + 1}`);

                    lossChart.data.labels = labels;
                    lossChart.data.datasets[0].data = losses;
                    lossChart.data.datasets[1].data = valLosses;
                    lossChart.update();

                    accuracyChart.data.labels = labels;
                    accuracyChart.data.datasets[0].data = accs;
                    accuracyChart.data.datasets[1].data = valAccs;
                    accuracyChart.update();

                    if (!st.is_training && currEpoch > 0) {
                        clearInterval(trainPollInterval);
                        btnTrain.disabled = false;
                        btnTrain.innerHTML = '<i class="fa-solid fa-play me-2"></i>Start Training';
                        showAlert('Model training finished successfully!', 'success');
                        loadDatasetInfo();
                    }
                }
            } catch (e) {
                console.error('Error polling status:', e);
            }
        }, 1500);
    }

    // Handle Generation
    btnGenerate.addEventListener('click', async () => {
        btnGenerate.disabled = true;
        btnGenerate.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i>Generating Music Notes...';

        const payload = {
            num_notes: parseInt(genNotesInput.value),
            temperature: parseFloat(genTempInput.value)
        };

        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) {
                showAlert(`Generated ${data.num_notes} notes with temperature ${data.temperature}!`, 'success');
                resultSection.classList.remove('d-none');
                resFilename.textContent = data.filename;
                btnDownload.href = `/download/${data.filename}`;
                btnDownload.setAttribute('download', data.filename);

                initPitchRollChart(data.notes || []);
                setupBrowserPlayer(`/play/${data.filename}`);
            } else {
                showAlert(data.error, 'danger');
            }
        } catch (err) {
            showAlert('Generation failed: ' + err, 'danger');
        } finally {
            btnGenerate.disabled = false;
            btnGenerate.innerHTML = '<i class="fa-solid fa-compact-disc me-2"></i>Generate Original Music';
        }
    });

    // Browser Tone.js Player
    async function setupBrowserPlayer(midiUrl) {
        playerStatus.textContent = 'Loading MIDI synthesis engine...';
        btnPlay.disabled = true;
        btnStop.disabled = true;

        try {
            // Parse MIDI from URL using @tonejs/midi
            const midi = await Midi.fromUrl(midiUrl);
            playerStatus.textContent = `Loaded '${midi.name || 'MIDI Track'}'. Ready to play.`;
            btnPlay.disabled = false;

            btnPlay.onclick = async () => {
                await Tone.start();
                if (isPlaying) return;

                if (!synth) {
                    synth = new Tone.PolySynth(Tone.Synth, {
                        oscillator: { type: 'triangle' },
                        envelope: { attack: 0.02, decay: 0.1, sustain: 0.3, release: 0.8 }
                    }).toDestination();
                }

                isPlaying = true;
                btnPlay.disabled = true;
                btnStop.disabled = false;
                playerStatus.textContent = '▶ Playing generated music audio...';

                // Schedule note events in Tone.Transport
                const now = Tone.now();
                midi.tracks.forEach(track => {
                    track.notes.forEach(note => {
                        synth.triggerAttackRelease(note.name, note.duration, note.time + now, note.velocity);
                    });
                });

                // Set stop timeout after track duration
                setTimeout(() => {
                    stopAudioPlayback();
                }, (midi.duration + 1) * 1000);
            };

            btnStop.onclick = () => {
                stopAudioPlayback();
            };

        } catch (err) {
            console.error('Tone.js synth error:', err);
            playerStatus.textContent = 'Direct audio synthesis unavailable. Use Download button to play in MIDI player.';
            btnPlay.disabled = true;
        }
    }

    function stopAudioPlayback() {
        if (synth) {
            synth.releaseAll();
        }
        isPlaying = false;
        btnPlay.disabled = false;
        btnStop.disabled = true;
        playerStatus.textContent = 'Playback stopped.';
    }

    // Init App
    initCharts();
    loadDatasetInfo();
});
