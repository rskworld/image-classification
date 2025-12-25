/*
===================================================================================
    Project: Image Classification Dataset
    Description: Advanced JavaScript with unique interactive features
    
    Author: Molla Samser
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Website: https://rskworld.in
    
    © 2025 RSK World. All rights reserved.
===================================================================================
*/

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initPreloader();
    initThemeToggle();
    initMobileMenu();
    initCounterAnimation();
    initTabsNavigation();
    initSmoothScrolling();
    initNavbarScrollEffect();
    initBackToTop();
    initParticles();
    initImageDemo();
    initGallery();
    initCharts();
    initDownloadSimulation();
    initScrollAnimations();
    
    // Console branding
    console.log('%c🖼️ Image Classification Dataset', 'font-size: 24px; font-weight: bold; color: #00d4ff;');
    console.log('%c© 2025 RSK World | https://rskworld.in', 'font-size: 12px; color: #a8b2c3;');
    console.log('%cDeveloped by Molla Samser | help@rskworld.in', 'font-size: 12px; color: #a8b2c3;');
});

/**
 * Preloader
 */
function initPreloader() {
    const preloader = document.getElementById('preloader');
    
    window.addEventListener('load', () => {
        setTimeout(() => {
            preloader.classList.add('hidden');
        }, 1500);
    });
    
    // Fallback: hide preloader after 3 seconds regardless
    setTimeout(() => {
        preloader.classList.add('hidden');
    }, 3000);
}

/**
 * Theme Toggle (Dark/Light)
 */
function initThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    const body = document.body;
    
    // Check saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    body.setAttribute('data-theme', savedTheme);
    
    themeToggle.addEventListener('click', () => {
        const currentTheme = body.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        body.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        
        // Re-initialize charts with new theme
        initCharts();
    });
}

/**
 * Mobile Menu
 */
function initMobileMenu() {
    const menuBtn = document.getElementById('mobileMenuBtn');
    const mobileMenu = document.getElementById('mobileMenu');
    
    if (menuBtn && mobileMenu) {
        menuBtn.addEventListener('click', () => {
            mobileMenu.classList.toggle('active');
            menuBtn.classList.toggle('active');
        });
        
        // Close menu when clicking a link
        mobileMenu.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                mobileMenu.classList.remove('active');
                menuBtn.classList.remove('active');
            });
        });
    }
}

/**
 * Counter Animation for Statistics
 */
function initCounterAnimation() {
    const counters = document.querySelectorAll('.stat-value[data-count]');
    
    const animateCounter = (counter) => {
        const target = parseInt(counter.getAttribute('data-count'));
        const duration = 2000;
        const increment = target / (duration / 16);
        let current = 0;
        
        const updateCounter = () => {
            current += increment;
            if (current < target) {
                counter.textContent = Math.floor(current).toLocaleString();
                requestAnimationFrame(updateCounter);
            } else {
                counter.textContent = target.toLocaleString();
            }
        };
        
        updateCounter();
    };
    
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    counters.forEach(counter => observer.observe(counter));
}

/**
 * Tabs Navigation for Code Examples
 */
function initTabsNavigation() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            button.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });
}

/**
 * Smooth Scrolling
 */
function initSmoothScrolling() {
    const navLinks = document.querySelectorAll('a[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href === '#') return;
            
            const targetElement = document.querySelector(href);
            
            if (targetElement) {
                e.preventDefault();
                const navbarHeight = document.querySelector('.navbar').offsetHeight;
                const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - navbarHeight - 20;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

/**
 * Navbar Scroll Effect
 */
function initNavbarScrollEffect() {
    const navbar = document.querySelector('.navbar');
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(10, 15, 26, 0.95)';
            navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
        } else {
            navbar.style.background = 'rgba(10, 15, 26, 0.8)';
            navbar.style.boxShadow = 'none';
        }
    });
}

/**
 * Back to Top Button
 */
function initBackToTop() {
    const backToTop = document.getElementById('backToTop');
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 500) {
            backToTop.classList.add('visible');
        } else {
            backToTop.classList.remove('visible');
        }
    });
    
    backToTop.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

/**
 * Floating Particles
 */
function initParticles() {
    const particlesContainer = document.getElementById('particles');
    if (!particlesContainer) return;
    
    const particleCount = 50;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.style.cssText = `
            position: absolute;
            width: ${Math.random() * 4 + 2}px;
            height: ${Math.random() * 4 + 2}px;
            background: rgba(0, 212, 255, ${Math.random() * 0.5 + 0.1});
            border-radius: 50%;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: floatParticle ${Math.random() * 10 + 10}s linear infinite;
            animation-delay: ${Math.random() * 5}s;
        `;
        particlesContainer.appendChild(particle);
    }
    
    // Add particle animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes floatParticle {
            0%, 100% { transform: translateY(0) translateX(0); opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { transform: translateY(-100vh) translateX(${Math.random() * 100 - 50}px); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
}

/**
 * Image Demo - Upload & Classification
 */
function initImageDemo() {
    const dropZone = document.getElementById('dropZone');
    const imageInput = document.getElementById('imageInput');
    const uploadPreview = document.getElementById('uploadPreview');
    const previewImage = document.getElementById('previewImage');
    const removePreview = document.getElementById('removePreview');
    const resultsContent = document.getElementById('resultsContent');
    const predictionResults = document.getElementById('predictionResults');
    const processingIndicator = document.getElementById('processingIndicator');
    const sampleBtns = document.querySelectorAll('.sample-btn');
    
    if (!dropZone) return;
    
    // Drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        });
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        });
    });
    
    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length) {
            handleFile(files[0]);
        }
    });
    
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });
    
    removePreview.addEventListener('click', resetDemo);
    
    // Sample buttons
    sampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const category = btn.getAttribute('data-category');
            simulatePrediction(category);
        });
    });
    
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Please upload an image file');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadPreview.style.display = 'block';
            dropZone.querySelector('.upload-content').style.display = 'none';
            
            // Simulate classification
            simulatePrediction();
        };
        reader.readAsDataURL(file);
    }
    
    function resetDemo() {
        uploadPreview.style.display = 'none';
        dropZone.querySelector('.upload-content').style.display = 'flex';
        resultsContent.style.display = 'block';
        predictionResults.style.display = 'none';
        imageInput.value = '';
    }
    
    function simulatePrediction(forcedCategory = null) {
        resultsContent.style.display = 'none';
        predictionResults.style.display = 'none';
        processingIndicator.style.display = 'flex';
        
        // Categories with icons
        const categories = [
            { name: 'Animals', icon: 'fa-dog' },
            { name: 'Vehicles', icon: 'fa-car' },
            { name: 'Nature', icon: 'fa-tree' },
            { name: 'Food', icon: 'fa-utensils' },
            { name: 'Buildings', icon: 'fa-building' },
            { name: 'Fashion', icon: 'fa-tshirt' },
            { name: 'Aircraft', icon: 'fa-plane' },
            { name: 'Sports', icon: 'fa-futbol' },
            { name: 'Electronics', icon: 'fa-laptop' },
            { name: 'Furniture', icon: 'fa-couch' }
        ];
        
        // Simulate processing delay
        setTimeout(() => {
            processingIndicator.style.display = 'none';
            predictionResults.style.display = 'block';
            
            // Generate predictions
            let predictions = categories.map(cat => ({
                ...cat,
                confidence: Math.random() * 30
            }));
            
            // Sort by confidence
            predictions.sort((a, b) => b.confidence - a.confidence);
            
            // If forced category, make it the top result
            if (forcedCategory) {
                const forcedIndex = predictions.findIndex(p => p.name === forcedCategory);
                if (forcedIndex > -1) {
                    predictions[forcedIndex].confidence = 90 + Math.random() * 9;
                    predictions.sort((a, b) => b.confidence - a.confidence);
                }
            } else {
                predictions[0].confidence = 85 + Math.random() * 14;
            }
            
            // Update top prediction
            const topClass = document.getElementById('topClass');
            const topConfidence = document.getElementById('topConfidence');
            const inferenceTime = document.getElementById('inferenceTime');
            
            topClass.textContent = predictions[0].name;
            topConfidence.textContent = predictions[0].confidence.toFixed(1) + '%';
            inferenceTime.textContent = (Math.random() * 0.3 + 0.1).toFixed(2) + 's';
            
            // Update all predictions
            const allPredictions = document.getElementById('allPredictions');
            allPredictions.innerHTML = predictions.slice(0, 5).map(pred => `
                <div class="prediction-item">
                    <span class="prediction-item-label">${pred.name}</span>
                    <div class="prediction-bar">
                        <div class="prediction-bar-fill" style="width: ${pred.confidence}%"></div>
                    </div>
                    <span class="prediction-item-value">${pred.confidence.toFixed(1)}%</span>
                </div>
            `).join('');
            
        }, 1500 + Math.random() * 1000);
    }
}

/**
 * Gallery with Filters
 */
function initGallery() {
    const galleryGrid = document.getElementById('galleryGrid');
    const filterBtns = document.querySelectorAll('.filter-btn');
    const prevPage = document.getElementById('prevPage');
    const nextPage = document.getElementById('nextPage');
    const currentPageEl = document.getElementById('currentPage');
    const totalPagesEl = document.getElementById('totalPages');
    
    if (!galleryGrid) return;
    
    const categories = [
        { name: 'animals', icon: 'fa-dog', color: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' },
        { name: 'vehicles', icon: 'fa-car', color: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' },
        { name: 'nature', icon: 'fa-tree', color: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' },
        { name: 'food', icon: 'fa-utensils', color: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)' },
        { name: 'buildings', icon: 'fa-building', color: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)' }
    ];
    
    let currentPage = 1;
    const itemsPerPage = 12;
    let currentFilter = 'all';
    let allItems = [];
    
    // Generate gallery items
    function generateItems() {
        allItems = [];
        for (let i = 0; i < 36; i++) {
            const cat = categories[i % categories.length];
            allItems.push({
                id: i,
                category: cat.name,
                icon: cat.icon,
                color: cat.color,
                name: `${cat.name.charAt(0).toUpperCase() + cat.name.slice(1)} ${i + 1}`
            });
        }
    }
    
    function renderGallery() {
        const filtered = currentFilter === 'all' 
            ? allItems 
            : allItems.filter(item => item.category === currentFilter);
        
        const totalPages = Math.ceil(filtered.length / itemsPerPage);
        const start = (currentPage - 1) * itemsPerPage;
        const end = start + itemsPerPage;
        const pageItems = filtered.slice(start, end);
        
        galleryGrid.innerHTML = pageItems.map(item => `
            <div class="gallery-item" data-category="${item.category}">
                <div class="gallery-item-image" style="background: ${item.color}">
                    <i class="fas ${item.icon}"></i>
                </div>
                <div class="gallery-item-overlay">${item.name}</div>
            </div>
        `).join('');
        
        currentPageEl.textContent = currentPage;
        totalPagesEl.textContent = totalPages || 1;
        
        prevPage.disabled = currentPage === 1;
        nextPage.disabled = currentPage >= totalPages;
    }
    
    // Filter buttons
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFilter = btn.getAttribute('data-filter');
            currentPage = 1;
            renderGallery();
        });
    });
    
    // Pagination
    prevPage.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderGallery();
        }
    });
    
    nextPage.addEventListener('click', () => {
        currentPage++;
        renderGallery();
    });
    
    generateItems();
    renderGallery();
}

/**
 * Charts with Chart.js
 */
function initCharts() {
    const isDark = document.body.getAttribute('data-theme') !== 'light';
    const textColor = isDark ? '#a8b2c3' : '#4b5563';
    const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
    
    // Category Distribution Chart
    const categoryCtx = document.getElementById('categoryChart');
    if (categoryCtx) {
        // Destroy existing chart if exists
        if (categoryCtx.chart) {
            categoryCtx.chart.destroy();
        }
        
        categoryCtx.chart = new Chart(categoryCtx, {
            type: 'doughnut',
            data: {
                labels: ['Animals', 'Vehicles', 'Nature', 'Food', 'Buildings', 'Others'],
                datasets: [{
                    data: [1500, 1200, 1000, 800, 900, 4600],
                    backgroundColor: [
                        '#667eea',
                        '#f093fb',
                        '#4facfe',
                        '#43e97b',
                        '#fa709a',
                        '#a8b2c3'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            color: textColor,
                            padding: 15,
                            font: { size: 11 }
                        }
                    }
                }
            }
        });
    }
    
    // Split Distribution Chart
    const splitCtx = document.getElementById('splitChart');
    if (splitCtx) {
        if (splitCtx.chart) {
            splitCtx.chart.destroy();
        }
        
        splitCtx.chart = new Chart(splitCtx, {
            type: 'bar',
            data: {
                labels: ['Training', 'Validation', 'Test'],
                datasets: [{
                    label: 'Images',
                    data: [7000, 1500, 1500],
                    backgroundColor: ['#00d4ff', '#7b68ee', '#ff6b6b'],
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: gridColor },
                        ticks: { color: textColor }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: textColor }
                    }
                }
            }
        });
    }
    
    // Performance Chart
    const performanceCtx = document.getElementById('performanceChart');
    if (performanceCtx) {
        if (performanceCtx.chart) {
            performanceCtx.chart.destroy();
        }
        
        performanceCtx.chart = new Chart(performanceCtx, {
            type: 'line',
            data: {
                labels: ['Epoch 1', 'Epoch 5', 'Epoch 10', 'Epoch 15', 'Epoch 20', 'Epoch 25', 'Epoch 30'],
                datasets: [{
                    label: 'Training Accuracy',
                    data: [45, 65, 78, 85, 90, 94, 97],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Validation Accuracy',
                    data: [40, 58, 72, 80, 85, 88, 92],
                    borderColor: '#7b68ee',
                    backgroundColor: 'rgba(123, 104, 238, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: textColor }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: gridColor },
                        ticks: { 
                            color: textColor,
                            callback: (value) => value + '%'
                        }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: textColor }
                    }
                }
            }
        });
    }
}

/**
 * Download Simulation
 */
function initDownloadSimulation() {
    const downloadBtn = document.getElementById('downloadBtn');
    const downloadProgress = document.getElementById('downloadProgress');
    const progressFill = document.getElementById('progressFill');
    const downloadPercent = document.getElementById('downloadPercent');
    
    if (!downloadBtn) return;
    
    downloadBtn.addEventListener('click', (e) => {
        // Check if it's a real download link
        const href = downloadBtn.getAttribute('href');
        if (href && href !== '#' && !href.includes('javascript:')) {
            // Let the real download happen
            return;
        }
        
        e.preventDefault();
        
        // Simulate download
        downloadBtn.style.display = 'none';
        downloadProgress.style.display = 'block';
        
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
                
                setTimeout(() => {
                    downloadProgress.style.display = 'none';
                    downloadBtn.style.display = 'flex';
                    showToast('Download simulation complete!');
                }, 500);
            }
            
            progressFill.style.width = progress + '%';
            downloadPercent.textContent = Math.floor(progress) + '%';
        }, 300);
    });
}

/**
 * Scroll Animations
 */
function initScrollAnimations() {
    const animatedElements = document.querySelectorAll('.feature-card, .category-card, .analytics-card, .testimonial-card');
    
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }, index * 50);
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    animatedElements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(30px)';
        element.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        observer.observe(element);
    });
}

/**
 * Copy Code to Clipboard
 */
function copyCode(tabId) {
    const codeElement = document.getElementById(tabId + 'Code');
    if (!codeElement) return;
    
    const code = codeElement.textContent;
    
    navigator.clipboard.writeText(code).then(() => {
        showToast('Code copied to clipboard!');
    }).catch(err => {
        console.error('Failed to copy:', err);
        showToast('Failed to copy code');
    });
}

/**
 * Show Toast Notification
 */
function showToast(message) {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');
    
    if (!toast) return;
    
    toastMessage.textContent = message;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

/**
 * Utility: Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Make copyCode globally available
window.copyCode = copyCode;
