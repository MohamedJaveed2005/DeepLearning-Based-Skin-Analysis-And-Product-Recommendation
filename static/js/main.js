(function() {
    'use strict';

    // Wait for DOM to be fully loaded
    document.addEventListener('DOMContentLoaded', function() {
        
        // --- Element references ---
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');
        const previewSection = document.getElementById('previewSection');
        const previewImage = document.getElementById('previewImage');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const resetBtn = document.getElementById('resetBtn');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const uploadSection = document.querySelector('.upload-section');

        // Results elements
        const resultImage = document.getElementById('resultImage');
        const skinTypeSpan = document.getElementById('skinType');
        const acneStatusSpan = document.getElementById('acneStatus');
        const skinConfidenceSpan = document.getElementById('skinConfidence');
        const acneConfidenceSpan = document.getElementById('acneConfidence');
        const tipsList = document.getElementById('tipsList');
        const productsGrid = document.getElementById('productsGrid');
        const noProductsMsg = document.getElementById('noProductsMessage');

        let selectedFile = null;

        // --- Helper: Reset to upload view ---
        function resetToUpload() {
            uploadArea.style.display = 'block';
            previewSection.style.display = 'none';
            loading.style.display = 'none';
            results.style.display = 'none';
            imageInput.value = '';
            selectedFile = null;
        }

        // --- Upload Area Events ---
        if (uploadArea) {
            uploadArea.addEventListener('click', () => imageInput.click());

            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = 'var(--color-green-deep)';
                uploadArea.style.background = 'rgba(143, 188, 148, 0.08)';
            });

            uploadArea.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = 'var(--color-green-soft)';
                uploadArea.style.background = 'rgba(255,255,255,0.5)';
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = 'var(--color-green-soft)';
                uploadArea.style.background = 'rgba(255,255,255,0.5)';
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFileSelect(files[0]);
                }
            });
        }

        // --- File input change ---
        if (imageInput) {
            imageInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFileSelect(e.target.files[0]);
                }
            });
        }

        // --- Handle selected file ---
        function handleFileSelect(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select a valid image file.');
                return;
            }

            selectedFile = file;

            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                uploadArea.style.display = 'none';
                previewSection.style.display = 'block';
                results.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        // --- Analyze button click ---
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', async () => {
                if (!selectedFile) {
                    alert('Please select an image first.');
                    return;
                }

                // Show loading
                previewSection.style.display = 'none';
                loading.style.display = 'block';
                results.style.display = 'none';

                const formData = new FormData();
                formData.append('image', selectedFile);

                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (data.error) {
                        alert('Error: ' + data.error);
                        resetToUpload();
                        return;
                    }

                    displayResults(data);

                } catch (error) {
                    console.error('Analysis error:', error);
                    alert('An error occurred during analysis. Please try again.');
                    resetToUpload();
                }
            });
        }

        // --- Display results ---
        function displayResults(data) {
            loading.style.display = 'none';
            results.style.display = 'block';

            // Result image
            if (resultImage) resultImage.src = data.image_path;

            // Skin type & acne
            if (skinTypeSpan) skinTypeSpan.textContent = data.skin_type || '--';
            if (acneStatusSpan) acneStatusSpan.textContent = data.acne_status || '--';

            // Confidence scores
            if (data.confidence) {
                if (skinConfidenceSpan) skinConfidenceSpan.textContent = data.confidence.skin || '';
                if (acneConfidenceSpan) acneConfidenceSpan.textContent = data.confidence.acne || '';
            }

            // Skincare tips
            if (tipsList) {
                tipsList.innerHTML = '';
                if (data.tips && data.tips.length) {
                    data.tips.forEach(tip => {
                        const li = document.createElement('li');
                        li.textContent = tip;
                        tipsList.appendChild(li);
                    });
                } else {
                    tipsList.innerHTML = '<li>Maintain a gentle skincare routine.</li>';
                }
            }

            // Product recommendations (with images)
            if (productsGrid) {
                productsGrid.innerHTML = '';
                if (data.products && data.products.length > 0) {
                    productsGrid.style.display = 'grid';
                    if (noProductsMsg) noProductsMsg.style.display = 'none';
                    
                    data.products.forEach(product => {
                        const card = document.createElement('div');
                        card.className = 'product-card';
                        
                        // Build image path
                        const imageUrl = product.image ? 
                            `/static/images/products/${product.image}` : 
                            '/static/images/products/placeholder.jpg';
                        
                        card.innerHTML = `
                            <div class="product-image">
                                <img src="${imageUrl}" alt="${product.name}" 
                                     onerror="this.src='/static/images/products/placeholder.jpg'">
                            </div>
                            <div class="product-info">
                                <div class="product-name">${product.name}</div>
                                <div class="product-brand">${product.brand}</div>
                                <div class="product-category">${product.category}</div>
                                <div class="product-price">${product.price}</div>
                                <div class="product-description">${product.description}</div>
                            </div>
                        `;
                        productsGrid.appendChild(card);
                    });
                } else {
                    productsGrid.style.display = 'none';
                    if (noProductsMsg) noProductsMsg.style.display = 'block';
                }
            }

            // Smooth scroll to results
            results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        // --- Reset button ---
        if (resetBtn) {
            resetBtn.addEventListener('click', resetToUpload);
        }

        // --- Debug: Log if any required element is missing ---
        const requiredIds = ['uploadArea', 'imageInput', 'previewSection', 'previewImage', 
                             'analyzeBtn', 'resetBtn', 'loading', 'results'];
        requiredIds.forEach(id => {
            if (!document.getElementById(id)) {
                console.warn(`Missing element with id: ${id}`);
            }
        });

        console.log('✅ Organic UI script loaded successfully.');
    });
})();