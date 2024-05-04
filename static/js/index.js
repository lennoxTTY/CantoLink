var stat_off = "（已停用）";
var stat_on = "（已啟用）";
var cm_list = "吖,啊,呀,噃,𠺢,咖,㗎,嘎,啩,吓,啦,咧,哩,囉,嗱,喎,呢,咋,喳,啫,唧";
var cs_list = "";
cmignore_list.disabled = true;
csignore_list.disabled = true;


common_stat.addEventListener('click', function(event){
  var common_stat = document.getElementById("common_stat");
  var cmignore_list = document.getElementById("cmignore_list");
  if (common_stat.innerHTML == stat_off){
    common_stat.innerHTML = stat_on;
    cmignore_list.value = cm_list;
  }else{
    common_stat.innerHTML = stat_off;
    cm_list = cmignore_list.value;
    cmignore_list.value = "";
  }
  cmignore_list.disabled = !cmignore_list.disabled;
});

custom_stat.addEventListener('click', function(event){
  var custom_stat = document.getElementById("custom_stat");
  var csignore_list = document.getElementById("csignore_list");
  if (custom_stat.innerHTML == stat_off){
    custom_stat.innerHTML = stat_on;
    csignore_list.value = cs_list;
  }else{
    custom_stat.innerHTML = stat_off;
    cs_list = csignore_list.value;
    csignore_list.value = "";
  }
  csignore_list.disabled = !csignore_list.disabled;
});

slide.addEventListener('input',function() {
  var slide = document.getElementById('slide'),
    sliderDiv = document.getElementById("sliderAmount");
    sliderDiv.innerHTML = slide.value;
}, false);

