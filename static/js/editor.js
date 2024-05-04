reload_btn = document.getElementById("reload_btn");
reload_btn.addEventListener("click",reload);

function reload(){
    video = document.getElementById("previewVideo");
    video.pause();
    video.currentTime = 0;
    video.load();
    video.play();
}