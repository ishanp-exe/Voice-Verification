// Client-side application controller for Customer Voice Authentication prototype
document.addEventListener('DOMContentLoaded', () => {
    // State management
    const state = {
        consentGiven: false,
        modelOnline: false,
        enrollRecorder: new WavRecorder(),
        verifyRecorder: new WavRecorder(),
        enrollBlobs: [], // Array of { id, name, durationSec, blob }
        currentRecordedEnrollBlob: null,
        verifyBlob: null,
        enrollMode: 'mic', // 'mic' | 'upload'
        verifyMode: 'mic',
        volunteers: []
    };

    // DOM Elements
    const elements = {
        modelStatusBadge: document.getElementById('model-status-badge'),
        consentCheckbox: document.getElementById('consent-checkbox'),
        volunteerFlowCards: document.querySelectorAll('.consent-dependent'),
        // Tabs
        tabBtns: document.querySelectorAll('.tab-btn'),
        tabPanels: document.querySelectorAll('.tab-panel'),
        // Enroll
        demoIdInput: document.getElementById('demo-id-input'),
        genIdBtn: document.getElementById('gen-id-btn'),
        enrollModeMicBtn: document.getElementById('enroll-mode-mic'),
        enrollModeUploadBtn: document.getElementById('enroll-mode-upload'),
        enrollMicBox: document.getElementById('enroll-mic-box'),
        enrollUploadBox: document.getElementById('enroll-upload-box'),
        enrollRecBtn: document.getElementById('enroll-rec-btn'),
        enrollRecTimer: document.getElementById('enroll-rec-timer'),
        enrollCanvas: document.getElementById('enroll-canvas'),
        enrollAudioPreview: document.getElementById('enroll-audio-preview'),
        enrollRecActions: document.getElementById('enroll-rec-actions'),
        enrollAddSampleBtn: document.getElementById('enroll-add-sample-btn'),
        enrollRetryBtn: document.getElementById('enroll-retry-btn'),
        enrollClearBtn: document.getElementById('enroll-clear-btn'),
        enrollFileInput: document.getElementById('enroll-file-input'),
        enrollDropzone: document.getElementById('enroll-dropzone'),
        enrollStagedContainer: document.getElementById('enroll-staged-container'),
        enrollSamplesCountBadge: document.getElementById('enroll-samples-count-badge'),
        enrollSamplesList: document.getElementById('enroll-samples-list'),
        enrollNoSamplesMsg: document.getElementById('enroll-no-samples-msg'),
        enrollSubmitBtn: document.getElementById('enroll-submit-btn'),
        enrollResultCard: document.getElementById('enroll-result-card'),
        // Verify
        verifyTargetSelect: document.getElementById('verify-target-select'),
        verifyModeMicBtn: document.getElementById('verify-mode-mic'),
        verifyModeUploadBtn: document.getElementById('verify-mode-upload'),
        verifyMicBox: document.getElementById('verify-mic-box'),
        verifyUploadBox: document.getElementById('verify-upload-box'),
        verifyRecBtn: document.getElementById('verify-rec-btn'),
        verifyRecTimer: document.getElementById('verify-rec-timer'),
        verifyCanvas: document.getElementById('verify-canvas'),
        verifyAudioPreview: document.getElementById('verify-audio-preview'),
        verifyRecActions: document.getElementById('verify-rec-actions'),
        verifyRetryBtn: document.getElementById('verify-retry-btn'),
        verifyClearBtn: document.getElementById('verify-clear-btn'),
        verifyFileInput: document.getElementById('verify-file-input'),
        verifyDropzone: document.getElementById('verify-dropzone'),
        verifySelectedPill: document.getElementById('verify-selected-pill'),
        thresholdSlider: document.getElementById('threshold-slider'),
        thresholdValueDisplay: document.getElementById('threshold-value-display'),
        verifySubmitBtn: document.getElementById('verify-submit-btn'),
        decisionCard: document.getElementById('decision-card'),
        // Volunteers Table
        volunteersTableBody: document.getElementById('volunteers-table-body'),
        volunteersEmptyNotice: document.getElementById('volunteers-empty-notice'),
        // Evaluation
        evalPathInput: document.getElementById('eval-path-input'),
        evalValidateBtn: document.getElementById('eval-validate-btn'),
        evalValidationNotice: document.getElementById('eval-validation-notice'),
        evalGenuineInput: document.getElementById('eval-genuine-input'),
        evalImposterInput: document.getElementById('eval-imposter-input'),
        evalSplitCheckbox: document.getElementById('eval-split-checkbox'),
        evalRunBtn: document.getElementById('eval-run-btn'),
        evalProgressWrap: document.getElementById('eval-progress-wrap'),
        evalProgressBar: document.getElementById('eval-progress-bar'),
        evalProgressText: document.getElementById('eval-progress-text'),
        evalResultsContainer: document.getElementById('eval-results-container'),
    };

    // Helper: Toast message
    function showToast(message, type = 'success') {
        const existing = document.querySelector('.toast-msg');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.className = `toast-msg toast-${type}`;
        toast.innerHTML = `<span>${type === 'success' ? '✅' : '⚠️'}</span> <span>${message}</span>`;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    // Helper: Generate Random ID
    function generateDemoId() {
        const rand = Math.floor(1000 + Math.random() * 9000);
        return `VOL-${rand}`;
    }

    // Helper: Canvas visualizer line
    function drawWaveform(canvas, dataArray) {
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        ctx.fillStyle = '#090D16';
        ctx.fillRect(0, 0, width, height);

        ctx.lineWidth = 2;
        ctx.strokeStyle = '#00F0FF';
        ctx.beginPath();

        const sliceWidth = width / dataArray.length;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * height) / 2;
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            x += sliceWidth;
        }

        ctx.lineTo(width, height / 2);
        ctx.stroke();
    }

    // Helper: Format timer
    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    // 1. Initial Health Check & Data Fetch
    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            const data = await res.json();
            if (data.status === 'ok' && data.model_loaded) {
                state.modelOnline = true;
                elements.modelStatusBadge.className = 'badge-pill status-online';
                elements.modelStatusBadge.innerHTML = '<span class="pulse-dot"></span> Engine Online (ECAPA-TDNN)';
            } else {
                elements.modelStatusBadge.innerHTML = '⚠️ Model Not Initialized';
            }
        } catch (err) {
            elements.modelStatusBadge.innerHTML = '❌ Engine Disconnected';
        }
    }

    // 2. Fetch Volunteers List
    async function fetchVolunteers() {
        try {
            const res = await fetch('/api/volunteers');
            const data = await res.json();
            state.volunteers = data.volunteers || [];
            renderVolunteersList();
        } catch (err) {
            console.error("Failed to load volunteers:", err);
        }
    }

    function renderVolunteersList() {
        const vols = state.volunteers;
        // Update Select in Verify Tab
        const prevSelected = elements.verifyTargetSelect.value;
        elements.verifyTargetSelect.innerHTML = '';
        if (vols.length === 0) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = '-- No enrolled profiles (Enroll in Step 1) --';
            elements.verifyTargetSelect.appendChild(opt);
            elements.verifySubmitBtn.disabled = true;
        } else {
            vols.forEach(v => {
                const opt = document.createElement('option');
                opt.value = v.demo_id;
                opt.textContent = `${v.demo_id} (${v.audio_duration_sec.toFixed(1)}s sample)`;
                elements.verifyTargetSelect.appendChild(opt);
            });
            if (prevSelected && vols.some(v => v.demo_id === prevSelected)) {
                elements.verifyTargetSelect.value = prevSelected;
            }
            elements.verifySubmitBtn.disabled = !state.consentGiven;
        }

        // Update Table
        elements.volunteersTableBody.innerHTML = '';
        if (vols.length === 0) {
            elements.volunteersEmptyNotice.style.display = 'block';
        } else {
            elements.volunteersEmptyNotice.style.display = 'none';
            vols.forEach(v => {
                const tr = document.createElement('tr');
                const cleanDate = (v.enrolled_at_utc || '').substring(0, 19).replace('T', ' ');
                tr.innerHTML = `
                    <td><strong style="color: var(--accent-cyan); font-family: monospace;">${v.demo_id}</strong></td>
                    <td style="color: var(--text-secondary);">${cleanDate} UTC</td>
                    <td>${v.audio_duration_sec.toFixed(1)} s (${v.num_enrollment_samples || 1} take${(v.num_enrollment_samples || 1) > 1 ? 's' : ''})</td>
                    <td><span class="badge-pill" style="font-size: 10px;">192-dim vector</span></td>
                    <td>
                        <button class="btn btn-danger btn-sm" data-id="${v.demo_id}" style="padding: 4px 10px; font-size: 11px;">
                            Purge
                        </button>
                    </td>
                `;
                elements.volunteersTableBody.appendChild(tr);
            });

            // Bind delete buttons
            elements.volunteersTableBody.querySelectorAll('button[data-id]').forEach(btn => {
                btn.addEventListener('click', () => deleteVolunteer(btn.getAttribute('data-id')));
            });
        }
    }

    async function deleteVolunteer(demoId) {
        if (!confirm(`Are you sure you want to permanently delete profile '${demoId}' and purge all local files?`)) {
            return;
        }
        try {
            const res = await fetch(`/api/volunteers/${demoId}`, { method: 'DELETE' });
            const data = await res.json();
            if (data.success) {
                showToast(`Profile '${demoId}' permanently deleted.`);
                await fetchVolunteers();
            } else {
                showToast(`Failed to delete profile: ${data.detail || data.error}`, 'error');
            }
        } catch (err) {
            showToast(`Deletion error: ${err.message}`, 'error');
        }
    }

    // Helper: Render Staged Enrollment Samples
    function renderStagedSamples() {
        if (!elements.enrollSamplesList) return;
        const total = state.enrollBlobs.length;
        if (elements.enrollSamplesCountBadge) {
            elements.enrollSamplesCountBadge.textContent = `${total} ready`;
        }

        elements.enrollSamplesList.innerHTML = '';
        if (total === 0) {
            if (elements.enrollNoSamplesMsg) {
                elements.enrollNoSamplesMsg.style.display = 'block';
                elements.enrollSamplesList.appendChild(elements.enrollNoSamplesMsg);
            }
        } else {
            if (elements.enrollNoSamplesMsg) elements.enrollNoSamplesMsg.style.display = 'none';
            state.enrollBlobs.forEach((item, idx) => {
                const row = document.createElement('div');
                row.className = 'staged-sample-row';
                row.style.cssText = 'display: flex; align-items: center; justify-content: space-between; padding: 6px 10px; background: rgba(30, 41, 59, 0.6); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); font-size: 12px;';

                const labelWrap = document.createElement('div');
                labelWrap.style.cssText = 'display: flex; align-items: center; gap: 8px; overflow: hidden;';
                const durText = item.durationSec > 0 ? ` (${item.durationSec.toFixed(1)}s)` : '';
                labelWrap.innerHTML = `<span style="color: var(--accent-cyan);">🎵</span> <span style="color: var(--text-primary); font-weight: 500; text-overflow: ellipsis; white-space: nowrap; overflow: hidden;">${item.name}${durText}</span>`;

                const removeBtn = document.createElement('button');
                removeBtn.type = 'button';
                removeBtn.innerHTML = '✖';
                removeBtn.title = 'Remove sample';
                removeBtn.style.cssText = 'background: transparent; border: none; color: #FB7185; cursor: pointer; font-size: 13px; padding: 2px 6px; border-radius: 4px;';
                removeBtn.addEventListener('click', () => {
                    state.enrollBlobs.splice(idx, 1);
                    renderStagedSamples();
                });

                row.appendChild(labelWrap);
                row.appendChild(removeBtn);
                elements.enrollSamplesList.appendChild(row);
            });
        }
        updateEnrollSubmitState();
    }

    function updateEnrollSubmitState() {
        const hasSamples = state.enrollBlobs.length > 0 || state.currentRecordedEnrollBlob !== null;
        elements.enrollSubmitBtn.disabled = !state.consentGiven || !hasSamples;
    }

    // Consent Checkbox listener update
    elements.consentCheckbox.addEventListener('change', (e) => {
        state.consentGiven = e.target.checked;
        elements.volunteerFlowCards.forEach(card => {
            if (state.consentGiven) {
                card.classList.remove('locked');
            } else {
                card.classList.add('locked');
            }
        });
        updateEnrollSubmitState();
        elements.verifySubmitBtn.disabled = !state.consentGiven || state.volunteers.length === 0;
    });

    // 4. Tab Switching
    elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            elements.tabBtns.forEach(b => b.classList.remove('active'));
            elements.tabPanels.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // 5. Enrollment Controls
    elements.demoIdInput.value = generateDemoId();
    elements.genIdBtn.addEventListener('click', () => {
        elements.demoIdInput.value = generateDemoId();
    });

    // Switch between mic and upload mode (Enroll)
    elements.enrollModeMicBtn.addEventListener('click', () => {
        state.enrollMode = 'mic';
        elements.enrollModeMicBtn.classList.add('active');
        elements.enrollModeUploadBtn.classList.remove('active');
        elements.enrollMicBox.style.display = 'block';
        elements.enrollUploadBox.style.display = 'none';
    });
    elements.enrollModeUploadBtn.addEventListener('click', () => {
        state.enrollMode = 'upload';
        elements.enrollModeUploadBtn.classList.add('active');
        elements.enrollModeMicBtn.classList.remove('active');
        elements.enrollUploadBox.style.display = 'block';
        elements.enrollMicBox.style.display = 'none';
    });

    // Mic recording (Enroll)
    elements.enrollRecBtn.addEventListener('click', async () => {
        if (!state.enrollRecorder.isRecording) {
            try {
                await state.enrollRecorder.start(
                    (data) => drawWaveform(elements.enrollCanvas, data),
                    (sec) => {
                        elements.enrollRecTimer.textContent = formatTime(sec);
                        if (sec >= 1.0) {
                            elements.enrollRecTimer.style.color = '#34D399';
                        }
                    }
                );
                elements.enrollRecBtn.className = 'rec-btn stop-rec';
                elements.enrollRecBtn.innerHTML = '⏹';
                elements.enrollAudioPreview.style.display = 'none';
                if (elements.enrollRecActions) elements.enrollRecActions.style.display = 'none';
            } catch (err) {
                showToast(err.message, 'error');
            }
        } else {
            const res = await state.enrollRecorder.stop();
            elements.enrollRecBtn.className = 'rec-btn start-rec';
            elements.enrollRecBtn.innerHTML = '🎙️';
            if (res) {
                state.currentRecordedEnrollBlob = { blob: res.blob, durationSec: res.durationSec };
                elements.enrollAudioPreview.src = URL.createObjectURL(res.blob);
                elements.enrollAudioPreview.style.display = 'block';
                if (elements.enrollRecActions) elements.enrollRecActions.style.display = 'flex';
                updateEnrollSubmitState();

                if (res.durationSec < 1.0) {
                    showToast(`Recording is ${res.durationSec.toFixed(1)}s. Minimum 1.0s recommended.`, 'error');
                } else {
                    showToast(`Captured ${res.durationSec.toFixed(1)}s sample. Click '+ Add to Enrollment Set' to include it.`);
                }
            }
        }
    });

    // Add to Enrollment Set handler
    if (elements.enrollAddSampleBtn) {
        elements.enrollAddSampleBtn.addEventListener('click', () => {
            if (!state.currentRecordedEnrollBlob) {
                showToast('No active recording to add. Record audio first.', 'error');
                return;
            }
            const sampleNum = state.enrollBlobs.length + 1;
            state.enrollBlobs.push({
                id: Date.now().toString(),
                name: `Voice Take #${sampleNum}`,
                durationSec: state.currentRecordedEnrollBlob.durationSec,
                blob: state.currentRecordedEnrollBlob.blob,
            });
            state.currentRecordedEnrollBlob = null;
            elements.enrollAudioPreview.src = '';
            elements.enrollAudioPreview.style.display = 'none';
            if (elements.enrollRecActions) elements.enrollRecActions.style.display = 'none';
            elements.enrollRecTimer.textContent = '00:00';
            elements.enrollRecTimer.style.color = 'var(--text-muted)';
            renderStagedSamples();
            showToast(`Voice Take #${sampleNum} added to enrollment set.`);
        });
    }

    // Retry and Clear handlers (Enroll)
    if (elements.enrollRetryBtn) {
        elements.enrollRetryBtn.addEventListener('click', async () => {
            state.currentRecordedEnrollBlob = null;
            elements.enrollAudioPreview.src = '';
            elements.enrollAudioPreview.style.display = 'none';
            if (elements.enrollRecActions) elements.enrollRecActions.style.display = 'none';
            elements.enrollRecTimer.textContent = '00:00';
            elements.enrollRecTimer.style.color = 'var(--text-muted)';
            updateEnrollSubmitState();
            // Trigger new recording
            elements.enrollRecBtn.click();
        });
    }

    if (elements.enrollClearBtn) {
        elements.enrollClearBtn.addEventListener('click', () => {
            state.currentRecordedEnrollBlob = null;
            elements.enrollAudioPreview.src = '';
            elements.enrollAudioPreview.style.display = 'none';
            if (elements.enrollRecActions) elements.enrollRecActions.style.display = 'none';
            elements.enrollRecTimer.textContent = '00:00';
            elements.enrollRecTimer.style.color = 'var(--text-muted)';
            updateEnrollSubmitState();
            showToast('Active recording cleared.');
        });
    }

    // File dropzone (Enroll)
    elements.enrollDropzone.addEventListener('click', () => elements.enrollFileInput.click());
    elements.enrollFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            Array.from(e.target.files).forEach((file, idx) => {
                state.enrollBlobs.push({
                    id: `${Date.now()}_${idx}`,
                    name: file.name,
                    durationSec: 0,
                    blob: file,
                });
            });
            elements.enrollFileInput.value = '';
            renderStagedSamples();
            showToast(`Added ${e.target.files.length} file(s) to enrollment set.`);
        }
    });

    // Submit Enrollment
    elements.enrollSubmitBtn.addEventListener('click', async () => {
        if (!state.consentGiven) {
            showToast("Informed consent is required.", 'error');
            return;
        }
        const demoId = elements.demoIdInput.value.trim();
        if (!demoId) {
            showToast("Please enter or generate a Demo Speaker ID.", 'error');
            return;
        }

        // Auto-stage active recording if user forgot to click "+ Add"
        if (state.enrollBlobs.length === 0 && state.currentRecordedEnrollBlob) {
            state.enrollBlobs.push({
                id: Date.now().toString(),
                name: 'Voice Take #1',
                durationSec: state.currentRecordedEnrollBlob.durationSec,
                blob: state.currentRecordedEnrollBlob.blob,
            });
            state.currentRecordedEnrollBlob = null;
            renderStagedSamples();
        }

        if (state.enrollBlobs.length === 0) {
            showToast("Please record or select at least one audio sample.", 'error');
            return;
        }

        const formData = new FormData();
        formData.append('demo_id', demoId);
        formData.append('consent', 'true');
        state.enrollBlobs.forEach((item, idx) => {
            formData.append('files', item.blob, item.name || `sample_${idx + 1}.wav`);
        });

        elements.enrollSubmitBtn.disabled = true;
        elements.enrollSubmitBtn.textContent = `Extracting & Averaging (${state.enrollBlobs.length} samples)...`;

        try {
            const res = await fetch('/api/volunteers/enroll', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (res.ok && data.success) {
                showToast(`Speaker '${demoId}' enrolled with ${data.num_samples} sample(s)!`);
                elements.enrollResultCard.style.display = 'block';
                elements.enrollResultCard.innerHTML = `
                    <div style="font-size: 13px; color: #34D399; font-weight: 700; margin-bottom: 4px;">✅ Voice Profile Registered (${data.num_samples} Take${data.num_samples > 1 ? 's' : ''})</div>
                    <div style="font-size: 12px; color: var(--text-secondary);">
                        Extracted and combined embeddings from <strong>${data.num_samples} separate recording(s)</strong> for <strong>${data.demo_id}</strong> (${data.duration_sec.toFixed(1)}s total audio).
                        Combined into one unit-normalized 192-dim vector. Raw audio discarded.
                    </div>
                `;
                // Reset state and generate new ID
                elements.demoIdInput.value = generateDemoId();
                state.enrollBlobs = [];
                state.currentRecordedEnrollBlob = null;
                elements.enrollAudioPreview.src = '';
                elements.enrollAudioPreview.style.display = 'none';
                if (elements.enrollRecActions) elements.enrollRecActions.style.display = 'none';
                elements.enrollRecTimer.textContent = '00:00';
                renderStagedSamples();
                await fetchVolunteers();
            } else {
                showToast(`Enrollment failed: ${data.detail || data.error}`, 'error');
            }
        } catch (err) {
            showToast(`Network/API error: ${err.message}`, 'error');
        } finally {
            elements.enrollSubmitBtn.disabled = false;
            elements.enrollSubmitBtn.textContent = "Enroll Voice Profile";
        }
    });

    // 6. Verification Controls
    elements.verifyModeMicBtn.addEventListener('click', () => {
        state.verifyMode = 'mic';
        elements.verifyModeMicBtn.classList.add('active');
        elements.verifyModeUploadBtn.classList.remove('active');
        elements.verifyMicBox.style.display = 'block';
        elements.verifyUploadBox.style.display = 'none';
    });
    elements.verifyModeUploadBtn.addEventListener('click', () => {
        state.verifyMode = 'upload';
        elements.verifyModeUploadBtn.classList.add('active');
        elements.verifyModeMicBtn.classList.remove('active');
        elements.verifyUploadBox.style.display = 'block';
        elements.verifyMicBox.style.display = 'none';
    });

    // Mic recording (Verify)
    elements.verifyRecBtn.addEventListener('click', async () => {
        if (!state.verifyRecorder.isRecording) {
            try {
                await state.verifyRecorder.start(
                    (data) => drawWaveform(elements.verifyCanvas, data),
                    (sec) => {
                        elements.verifyRecTimer.textContent = formatTime(sec);
                        if (sec >= 1.0) {
                            elements.verifyRecTimer.style.color = '#34D399';
                        }
                    }
                );
                elements.verifyRecBtn.className = 'rec-btn stop-rec';
                elements.verifyRecBtn.innerHTML = '⏹';
                elements.verifyAudioPreview.style.display = 'none';
                if (elements.verifyRecActions) elements.verifyRecActions.style.display = 'none';
            } catch (err) {
                showToast(err.message, 'error');
            }
        } else {
            const res = await state.verifyRecorder.stop();
            elements.verifyRecBtn.className = 'rec-btn start-rec';
            elements.verifyRecBtn.innerHTML = '🎙️';
            if (res) {
                state.verifyBlob = res.blob;
                elements.verifyAudioPreview.src = URL.createObjectURL(res.blob);
                elements.verifyAudioPreview.style.display = 'block';
                if (elements.verifyRecActions) elements.verifyRecActions.style.display = 'flex';
                if (res.durationSec < 1.0) {
                    showToast(`Candidate sample is ${res.durationSec.toFixed(1)}s (min 1.0s recommended).`, 'error');
                } else {
                    showToast(`Captured ${res.durationSec.toFixed(1)}s candidate sample.`);
                }
            }
        }
    });

    // Retry and Clear handlers (Verify)
    if (elements.verifyRetryBtn) {
        elements.verifyRetryBtn.addEventListener('click', async () => {
            state.verifyBlob = null;
            elements.verifyAudioPreview.src = '';
            elements.verifyAudioPreview.style.display = 'none';
            if (elements.verifyRecActions) elements.verifyRecActions.style.display = 'none';
            elements.verifyRecTimer.textContent = '00:00';
            elements.verifyRecTimer.style.color = 'var(--text-muted)';
            // Trigger new recording
            elements.verifyRecBtn.click();
        });
    }

    if (elements.verifyClearBtn) {
        elements.verifyClearBtn.addEventListener('click', () => {
            state.verifyBlob = null;
            elements.verifyAudioPreview.src = '';
            elements.verifyAudioPreview.style.display = 'none';
            if (elements.verifyRecActions) elements.verifyRecActions.style.display = 'none';
            elements.verifyRecTimer.textContent = '00:00';
            elements.verifyRecTimer.style.color = 'var(--text-muted)';
            showToast('Candidate recording cleared.');
        });
    }

    // File dropzone (Verify)
    elements.verifyDropzone.addEventListener('click', () => elements.verifyFileInput.click());
    elements.verifyFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            state.verifyBlob = file;
            elements.verifySelectedPill.style.display = 'inline-flex';
            elements.verifySelectedPill.textContent = `📁 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        }
    });

    // Threshold slider display
    elements.thresholdSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value).toFixed(2);
        elements.thresholdValueDisplay.textContent = val;
    });

    // Submit Verification
    elements.verifySubmitBtn.addEventListener('click', async () => {
        if (!state.consentGiven) {
            showToast("Informed consent is required.", 'error');
            return;
        }
        const targetId = elements.verifyTargetSelect.value;
        if (!targetId) {
            showToast("Please select an enrolled speaker ID.", 'error');
            return;
        }
        if (!state.verifyBlob) {
            showToast("Please record or upload a candidate voice sample.", 'error');
            return;
        }

        const threshold = parseFloat(elements.thresholdSlider.value);
        const formData = new FormData();
        formData.append('demo_id', targetId);
        formData.append('threshold', threshold.toString());
        formData.append('consent', 'true');
        formData.append('file', state.verifyBlob, 'candidate.wav');

        elements.verifySubmitBtn.disabled = true;
        elements.verifySubmitBtn.textContent = "Computing Cosine Similarity...";

        try {
            const res = await fetch('/api/volunteers/verify', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (res.ok && data.success) {
                renderDecisionCard(data);
            } else {
                showToast(`Verification failed: ${data.detail || data.error}`, 'error');
            }
        } catch (err) {
            showToast(`Network/API error: ${err.message}`, 'error');
        } finally {
            elements.verifySubmitBtn.disabled = false;
            elements.verifySubmitBtn.textContent = "Verify Candidate Voice";
        }
    });

    function renderDecisionCard(result) {
        const card = elements.decisionCard;
        card.style.display = 'block';
        card.className = `decision-card ${result.is_match ? 'match' : 'nomatch'}`;

        const margin = result.similarity - result.threshold;
        const normSimPct = Math.max(0, Math.min(100, ((result.similarity + 1) / 2) * 100));
        const normThreshPct = Math.max(0, Math.min(100, ((result.threshold + 1) / 2) * 100));

        card.innerHTML = `
            <div class="verdict-header">
                <div class="verdict-title">
                    ${result.is_match ? '✅ VERDICT: MATCH' : '❌ VERDICT: NO MATCH'}
                </div>
                <div style="font-size: 12px; font-weight: 600; text-transform: uppercase;">
                    Target: ${result.demo_id}
                </div>
            </div>

            <div class="metrics-row">
                <div class="metric-tile">
                    <div class="metric-tile-label">Cosine Similarity</div>
                    <div class="metric-tile-val" style="color: ${result.is_match ? '#34D399' : '#FB7185'};">
                        ${result.similarity.toFixed(4)}
                    </div>
                </div>
                <div class="metric-tile">
                    <div class="metric-tile-label">Operational Threshold</div>
                    <div class="metric-tile-val">
                        ${result.threshold.toFixed(4)}
                    </div>
                </div>
                <div class="metric-tile">
                    <div class="metric-tile-label">Decision Margin</div>
                    <div class="metric-tile-val" style="color: ${margin >= 0 ? '#34D399' : '#FB7185'};">
                        ${margin >= 0 ? '+' : ''}${margin.toFixed(4)}
                    </div>
                </div>
            </div>

            <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 4px; display: flex; justify-content: space-between;">
                <span>Score Scale (-1.0 to +1.0)</span>
                <span>Pin: Threshold (${result.threshold.toFixed(2)})</span>
            </div>
            <div class="gauge-wrapper">
                <div class="gauge-fill" style="width: ${normSimPct}%;"></div>
                <div class="gauge-threshold-pin" style="left: ${normThreshPct}%;"></div>
            </div>

            <div class="decision-explanation">
                ${result.explanation}
            </div>
        `;
    }

    // 7. LibriSpeech Evaluation Controls
    elements.evalValidateBtn.addEventListener('click', async () => {
        const path = elements.evalPathInput.value.trim();
        if (!path) {
            elements.evalValidationNotice.style.display = 'block';
            elements.evalValidationNotice.className = 'disclaimer-banner';
            elements.evalValidationNotice.innerHTML = "⚠️ Please enter an external LibriSpeech directory path.";
            return;
        }

        try {
            const res = await fetch('/api/evaluation/validate-dataset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: path })
            });
            const data = await res.json();
            elements.evalValidationNotice.style.display = 'block';
            if (data.is_valid) {
                elements.evalValidationNotice.className = 'disclaimer-banner';
                elements.evalValidationNotice.style.background = 'rgba(16, 185, 129, 0.1)';
                elements.evalValidationNotice.style.borderLeftColor = 'var(--accent-emerald)';
                elements.evalValidationNotice.style.color = '#A7F3D0';
                elements.evalValidationNotice.innerHTML = `
                    <strong>✅ Valid Corpus Detected:</strong> Found <strong>${data.num_speakers}</strong> speakers 
                    with <strong>${data.num_recordings}</strong> total audio recordings. Ready for trial generation.
                `;
            } else {
                elements.evalValidationNotice.className = 'disclaimer-banner';
                elements.evalValidationNotice.style.background = 'rgba(244, 63, 94, 0.1)';
                elements.evalValidationNotice.style.borderLeftColor = 'var(--accent-rose)';
                elements.evalValidationNotice.style.color = '#FECDD3';
                elements.evalValidationNotice.innerHTML = `❌ ${data.message}`;
            }
        } catch (err) {
            showToast(`Validation request failed: ${err.message}`, 'error');
        }
    });

    elements.evalRunBtn.addEventListener('click', async () => {
        const path = elements.evalPathInput.value.trim();
        if (!path) {
            showToast("Please provide the external LibriSpeech path first.", 'error');
            return;
        }

        const numGenuine = parseInt(elements.evalGenuineInput.value, 10) || 20;
        const numImposter = parseInt(elements.evalImposterInput.value, 10) || 20;
        const enableSplit = elements.evalSplitCheckbox.checked;

        elements.evalRunBtn.disabled = true;
        elements.evalProgressWrap.style.display = 'block';
        elements.evalProgressBar.style.width = '30%';
        elements.evalProgressText.textContent = "Loading audio and extracting embeddings...";
        elements.evalResultsContainer.style.display = 'none';

        try {
            const res = await fetch('/api/evaluation/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    path: path,
                    num_genuine: numGenuine,
                    num_imposter: numImposter,
                    enable_split: enableSplit
                })
            });

            elements.evalProgressBar.style.width = '100%';
            const data = await res.json();

            if (res.ok && data.success) {
                renderEvaluationResults(data);
            } else {
                showToast(`Evaluation error: ${data.detail || data.error}`, 'error');
            }
        } catch (err) {
            showToast(`Evaluation failed: ${err.message}`, 'error');
        } finally {
            elements.evalRunBtn.disabled = false;
            elements.evalProgressWrap.style.display = 'none';
        }
    });

    function renderEvaluationResults(data) {
        const cont = elements.evalResultsContainer;
        cont.style.display = 'block';

        let badgeHeader = '';
        if (data.protocol === 'calibration_split') {
            badgeHeader = `
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: var(--radius-md); padding: 14px 18px; margin-bottom: 18px; color: #A7F3D0; font-size: 13px;">
                    <strong>✅ Disciplined Calibration & Test Protocol Completed:</strong><br>
                    Operating threshold was calibrated on disjoint Dev partition: <strong>τ = ${data.calibrated_threshold.toFixed(4)}</strong>.<br>
                    Final metrics below were evaluated strictly on the unseen Test partition. We do not tune or report an 'EER' on the test set.
                </div>
            `;
        } else {
            badgeHeader = `
                <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: var(--radius-md); padding: 14px 18px; margin-bottom: 18px; color: #FDE68A; font-size: 13px;">
                    <strong>⚠️ Development Preview Mode (Unsplit Data):</strong><br>
                    Dataset was too small for a split or split was disabled. Metrics and EER threshold are descriptive development indicators only.
                </div>
            `;
        }

        // Skipped trials table if any
        let skippedHtml = '';
        if (data.skipped_trials && data.skipped_trials.length > 0) {
            skippedHtml = `
                <div style="margin-top: 14px; padding: 12px; background: rgba(244, 63, 94, 0.08); border-radius: var(--radius-md); font-size: 12px;">
                    <strong style="color: #FB7185;">⚠️ ${data.skipped_trials.length} trial(s) were unreadable or skipped:</strong>
                    <ul style="margin-top: 6px; padding-left: 18px; color: var(--text-secondary);">
                        ${data.skipped_trials.map(s => `<li>${s.Speakers} (${s.Files}): ${s.Error || s.Reason}</li>`).join('')}
                    </ul>
                </div>
            `;
        // Trial breakdown & metadata display
        let metaDetailsHtml = '';
        if (data.trial_breakdown) {
            const calGen = data.trial_breakdown.calibration_evaluated?.genuine ?? 'N/A';
            const calImp = data.trial_breakdown.calibration_evaluated?.imposter ?? 'N/A';
            const testGen = data.trial_breakdown.test_evaluated?.genuine ?? data.trial_breakdown.evaluated?.genuine ?? 'N/A';
            const testImp = data.trial_breakdown.test_evaluated?.imposter ?? data.trial_breakdown.evaluated?.imposter ?? 'N/A';
            const calSeed = data.seeds?.calibration_seed ?? data.seeds?.seed ?? 'N/A';
            const testSeed = data.seeds?.evaluation_seed ?? 'N/A';
            const cachedCount = data.cache_stats?.unique_recordings_cached ?? 'N/A';

            metaDetailsHtml = `
                <div style="margin-top: 16px; padding: 12px 14px; background: rgba(30, 41, 59, 0.4); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); font-size: 12px; color: var(--text-secondary); line-height: 1.6;">
                    <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">Trial Distribution & Reproducibility Parameters:</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                        <div>• Calibration Partition: <strong>${calGen}</strong> genuine, <strong>${calImp}</strong> imposter (seed ${calSeed})</div>
                        <div>• Held-Out Evaluation: <strong>${testGen}</strong> genuine, <strong>${testImp}</strong> imposter (seed ${testSeed})</div>
                        <div>• Inference Optimization: <strong>${cachedCount}</strong> unique recordings cached</div>
                        <div>• Evaluation Total: <strong>${data.total_trials}</strong> completed trial pairs</div>
                    </div>
                </div>
            `;
        }

        const disclaimerHtml = `
            <div style="margin-top: 14px; font-size: 11px; color: var(--text-muted); font-style: italic; border-top: 1px solid var(--border-subtle); padding-top: 10px;">
                ⚠️ <strong>Acoustic Domain Boundary:</strong> ${data.disclaimer || 'Evaluated on clean audiobook speech (LibriSpeech test-clean). Does not establish performance on telephone banking audio.'}
            </div>
        `;

        cont.innerHTML = `
            ${badgeHeader}
            <div class="metrics-row" style="grid-template-columns: repeat(4, 1fr);">
                <div class="metric-tile">
                    <div class="metric-tile-label">Operating Threshold</div>
                    <div class="metric-tile-val" style="color: var(--accent-cyan);">
                        ${data.calibrated_threshold.toFixed(4)}
                    </div>
                </div>
                <div class="metric-tile">
                    <div class="metric-tile-label">Test False Accept Rate</div>
                    <div class="metric-tile-val" style="color: #FB7185;">
                        ${(data.test_far * 100).toFixed(2)}%
                    </div>
                </div>
                <div class="metric-tile">
                    <div class="metric-tile-label">Test False Reject Rate</div>
                    <div class="metric-tile-val" style="color: #38BDF8;">
                        ${(data.test_frr * 100).toFixed(2)}%
                    </div>
                </div>
                <div class="metric-tile">
                    <div class="metric-tile-label">Test Accuracy</div>
                    <div class="metric-tile-val" style="color: #34D399;">
                        ${(data.test_accuracy * 100).toFixed(2)}%
                    </div>
                </div>
            </div>
            ${metaDetailsHtml}
            ${skippedHtml}
            ${disclaimerHtml}
        `;
    }

    // Initialize
    checkHealth();
    fetchVolunteers();
});
