window.addEventListener("DOMContentLoaded", () => {
    const face = document.getElementById("kabuchan-face");
    const bubble = document.getElementById("speech-bubble");

    
    face.style.backgroundImage = "url('/static/kabuchan_pngs/kabuchan_thinking_touka.png')";

    
    setTimeout(() => {
        face.style.backgroundImage = "url('/static/kabuchan_pngs/kabuchan_thumsup_touka.png')";
        bubble.style.display = "block";
        bubble.style.opacity = 0;
        bubble.style.transition = "opacity 1.0s";
        setTimeout(() => {
            bubble.style.opacity = 1;
        }, 50);
    }, 3000);
});