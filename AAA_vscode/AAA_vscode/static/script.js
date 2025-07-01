class TrafficSignApp {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
    }

    setupEventListeners() {
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');

        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        uploadArea.addEventListener('click', () => fileInput.click());
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('uploadArea');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
        });

        uploadArea.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        if (!this.validateFile(file)) {
            this.showError('Please select a valid image file (JPG, PNG, JPEG)');
            return;
        }

        this.showProcessing();
        this.uploadFile(file);
    }

    validateFile(file) {
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        return allowedTypes.includes(file.type);
    }

    showProcessing() {
        document.getElementById('uploadSection').classList.add('hidden');
        document.getElementById('processingSection').classList.remove('hidden');
        document.getElementById('resultsSection').classList.add('hidden');

        this.simulateProgress();
    }

    simulateProgress() {
        const statusElement = document.getElementById('processingStatus');
        const progressFill = document.getElementById('progressFill');
        
        const steps = [
            { text: 'Loading image...', progress: 10 },
            { text: 'Running YOLO detection...', progress: 40 },
            { text: 'Cropping detected regions...', progress: 60 },
            { text: 'Classifying with CNN...', progress: 80 },
            { text: 'Finalizing results...', progress: 100 }
        ];

        let currentStep = 0;
        
        const updateProgress = () => {
            if (currentStep < steps.length) {
                const step = steps[currentStep];
                statusElement.textContent = step.text;
                progressFill.style.width = step.progress + '%';
                currentStep++;
                setTimeout(updateProgress, 800);
            }
        };

        updateProgress();
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const startTime = Date.now();
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            const processingTime = Date.now() - startTime;

            setTimeout(() => {
                if (result.success) {
                    this.showResults(result, processingTime);
                } else {
                    this.showError(result.error);
                }
            }, 2000); // Wait for progress animation to complete

        } catch (error) {
            setTimeout(() => {
                this.showError('An error occurred while processing the image');
            }, 2000);
        }
    }

    showResults(data, processingTime) {
        document.getElementById('processingSection').classList.add('hidden');
        document.getElementById('resultsSection').classList.remove('hidden');

        // Display images
        const originalImage = document.getElementById('originalImage');
        const annotatedImage = document.getElementById('annotatedImage');
        
        originalImage.src = `/uploads/${data.original_filename}`;
        annotatedImage.src = `/results/${data.annotated_image}`;

        // Display statistics
        document.getElementById('detectionCount').textContent = data.detections_count;
        document.getElementById('processingTime').textContent = `${processingTime}ms`;

        // Display classifications
        this.displayClassifications(data.classifications);
    }

    displayClassifications(classifications) {
        const grid = document.getElementById('classificationsGrid');
        grid.innerHTML = '';

        classifications.forEach(classification => {
            const item = document.createElement('div');
            item.className = 'classification-item';
            
            item.innerHTML = `
                <img src="/results/${classification.crop_filename}" alt="${classification.class_name}">
                <div class="classification-label">${classification.class_name}</div>
                <div class="classification-confidence">${(classification.confidence * 100).toFixed(1)}%</div>
            `;
            
            grid.appendChild(item);
        });
    }

    showError(message) {
        document.getElementById('processingSection').classList.add('hidden');
        document.getElementById('resultsSection').classList.add('hidden');
        document.getElementById('uploadSection').classList.remove('hidden');
        
        alert(`Error: ${message}`);
    }
}

function resetApplication() {
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('processingSection').classList.add('hidden');
    document.getElementById('uploadSection').classList.remove('hidden');
    
    // Clear file input
    document.getElementById('fileInput').value = '';
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TrafficSignApp();
});
