window.addEventListener("DOMContentLoaded", () => {
  setTimeout(() => {
    const face = document.querySelector(".kabuchan-face");
    const faceClass = face.dataset.faceClass;
    const mood = face.dataset.mood;
    const score = face.dataset.score;

    // 表情を変更
    face.className = "face " + faceClass;

    // メッセージを書き換える
    const message = document.getElementById("kabuchan-message");
    message.innerHTML = `
      <p class="speech-title">今日の市場の様子は・・・</p>
      <ul>
        <li><span class="stock-name">${mood}</span></li>
        <li>感情スコア：${score}/5</li>
      </ul>
    `;
  }, 2000);
});