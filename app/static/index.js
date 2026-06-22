document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const dropzoneCard = document.getElementById('dropzone-card');
    const loadingCard = document.getElementById('loading-card');
    const resultCard = document.getElementById('result-card');
    const beforeImg = document.getElementById('before-img');
    const afterImg = document.getElementById('after-img');
    const resetBtn = document.getElementById('reset-btn');
    const downloadBtn = document.getElementById('download-btn');
    const loaderStatus = document.getElementById('loader-status');
    
    // Slider Elements
    const sliderContainer = document.getElementById('slider-container');
    const overlayContainer = document.getElementById('overlay-container');
    const sliderHandle = document.getElementById('slider-handle');

    let isDragging = false;
    let originalObjectUrl = null;
    let styledObjectUrl = null;

    // Load Status Text rotation to make loading feel dynamic
    const statusMessages = [
        "Analyzing human facial structures...",
        "Applying convolutional downsampling...",
        "Extracting multi-layer geometric patches...",
        "Stylizing details with generator model...",
        "Rendering outlines & anime cells...",
        "Finalizing style transfer..."
    ];
    let statusInterval = null;

    function startStatusRotation() {
        let index = 0;
        loaderStatus.textContent = statusMessages[index];
        statusInterval = setInterval(() => {
            index = (index + 1) % statusMessages.length;
            loaderStatus.textContent = statusMessages[index];
        }, 2200);
    }

    function stopStatusRotation() {
        if (statusInterval) {
            clearInterval(statusInterval);
            statusInterval = null;
        }
    }

    // Drag and Drop Listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropArea.classList.remove('dragover');
        }, false);
    });

    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    dropArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // File Processing
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Error: Please upload a valid image file (PNG, JPG).');
            return;
        }
        
        // 10MB limit
        if (file.size > 10 * 1024 * 1024) {
            alert('Error: Image size exceeds 10MB.');
            return;
        }

        // Display loading
        dropzoneCard.classList.add('hidden');
        loadingCard.classList.remove('hidden');
        startStatusRotation();

        // Create Object URL for preview
        originalObjectUrl = URL.createObjectURL(file);
        beforeImg.src = originalObjectUrl;

        // Call FastAPI translate endpoint
        const formData = new FormData();
        formData.append('file', file);

        fetch('/api/translate', {
            method: 'POST',
            body: formData
        })
        .then(async response => {
            if (!response.ok) {
                try {
                    const errData = await response.json();
                    if (errData && errData.detail) {
                        throw new Error(errData.detail);
                    }
                } catch (e) {
                    if (e.message && !e.message.includes('JSON')) {
                        throw e;
                    }
                }
                throw new Error('Server returned error status: ' + response.status);
            }
            return response.blob();
        })
        .then(blob => {
            styledObjectUrl = URL.createObjectURL(blob);
            afterImg.src = styledObjectUrl;
            
            // Set download link href
            downloadBtn.href = styledObjectUrl;

            // Wait for image loading to avoid visual pop
            afterImg.onload = () => {
                loadingCard.classList.add('hidden');
                resultCard.classList.remove('hidden');
                stopStatusRotation();
                resetSlider();
            };
        })
        .catch(error => {
            console.error('Translation error:', error);
            alert(error.message || 'Image translation failed. Check backend console logs.');
            resetApp();
        });
    }

    // Reset Functions
    function resetApp() {
        stopStatusRotation();
        dropzoneCard.classList.remove('hidden');
        loadingCard.classList.add('hidden');
        resultCard.classList.add('hidden');
        fileInput.value = '';
        
        // Revoke Object URLs to prevent memory leaks
        if (originalObjectUrl) {
            URL.revokeObjectURL(originalObjectUrl);
            originalObjectUrl = null;
        }
        if (styledObjectUrl) {
            URL.revokeObjectURL(styledObjectUrl);
            styledObjectUrl = null;
        }
    }

    resetBtn.addEventListener('click', resetApp);

    // Slider Comparison Logic
    function resetSlider() {
        overlayContainer.style.width = '50%';
        sliderHandle.style.left = '50%';
    }

    function moveSlider(clientX) {
        const rect = sliderContainer.getBoundingClientRect();
        const position = clientX - rect.left;
        let percentage = (position / rect.width) * 100;

        // Bounds constraints
        if (percentage < 0) percentage = 0;
        if (percentage > 100) percentage = 100;

        overlayContainer.style.width = `${percentage}%`;
        sliderHandle.style.left = `${percentage}%`;
    }

    // Mouse events
    sliderHandle.addEventListener('mousedown', (e) => {
        isDragging = true;
        e.preventDefault();
    });

    window.addEventListener('mouseup', () => {
        isDragging = false;
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        e.preventDefault(); // Prevents text selection / drag behavior
        moveSlider(e.clientX);
    });

    // Touch events for mobile screens
    sliderHandle.addEventListener('touchstart', (e) => {
        isDragging = true;
        e.preventDefault(); // Prevents drag-scroll activation
    }, { passive: false });

    window.addEventListener('touchend', () => {
        isDragging = false;
    });

    window.addEventListener('touchmove', (e) => {
        if (!isDragging) return;
        e.preventDefault(); // Prevents screen scrolling while sliding
        if (e.touches.length > 0) {
            moveSlider(e.touches[0].clientX);
        }
    }, { passive: false });
});
