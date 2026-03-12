let currentSlideIndex = 1;
const slides = document.querySelectorAll('.slide');
const totalSlides = slides.length;

const progressBar = document.getElementById('progressBar');
const currentSlideDisplay = document.getElementById('currentSlide');
const totalSlidesDisplay = document.getElementById('totalSlides');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');

function goToSlide(slideNumber) {
    if (slideNumber < 1 || slideNumber > totalSlides) return;

    const currentSlide = document.querySelector('.slide.active');
    const nextSlide = document.querySelector(`.slide[data-slide="${slideNumber}"]`);

    if (currentSlide) {
        currentSlide.classList.remove('active');
        currentSlide.classList.add('exit');
        setTimeout(() => currentSlide.classList.remove('exit'), 600);
    }

    if (nextSlide) {
        setTimeout(() => {
            nextSlide.classList.add('active');
            restartAnimations(nextSlide);
            
            // Перерендеринг MathJax для нового слайда
            if (window.MathJax) {
                window.MathJax.typesetPromise([nextSlide]);
            }
        }, 100);
    }

    currentSlideIndex = slideNumber;
    updateUI();
}

function nextSlide() {
    if (currentSlideIndex < totalSlides) goToSlide(currentSlideIndex + 1);
}

function prevSlide() {
    if (currentSlideIndex > 1) goToSlide(currentSlideIndex - 1);
}

function updateUI() {
    currentSlideDisplay.textContent = currentSlideIndex;
    totalSlidesDisplay.textContent = totalSlides;
    progressBar.style.width = `${(currentSlideIndex / totalSlides) * 100}%`;
    prevBtn.disabled = currentSlideIndex === 1;
    nextBtn.disabled = currentSlideIndex === totalSlides;
}

function restartAnimations(slide) {
    const animated = slide.querySelectorAll('.animate-in');
    animated.forEach(el => {
        el.style.animation = 'none';
        void el.offsetWidth;
        el.style.animation = null;
    });
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ') nextSlide();
    if (e.key === 'ArrowLeft') prevSlide();
});

prevBtn.addEventListener('click', prevSlide);
nextBtn.addEventListener('click', nextSlide);

const restartBtn = document.getElementById('restartBtn');
const fireEffect = document.getElementById('fireEffect');

if (restartBtn && fireEffect) {
    restartBtn.addEventListener('click', () => {
        // Активуємо полум'я
        fireEffect.classList.add('active');
        
        // Змінюємо текст кнопки для фінального акценту
        restartBtn.textContent = "Курс розпочато!";
        restartBtn.style.borderColor = "#ffcf33";
    });
}

// Ініціалізація
updateUI();
