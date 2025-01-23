function getConceptColor(index) {
    if (index === 0) {
        return "#ff0000ff"
    } else if (index === 1) {
        return "#00ff00ff"
    } else if (index === 2) {
        return "#0000ffff"
    } else if (index === 3) {
        return "#ff9900ff"
    } else {
        return "#f0f0f0"
    }
}

function showPlot(index) {
    let tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => tab.classList.remove('active-tab'));
    tabs.forEach(tab => tab.style.background = "#f0f0f0");
    tabs.forEach(tab => tab.style.border = "2px solid #ccc")
    tabs[index].classList.add('active-tab');
    tabs[index].style.background = getConceptColor(index);

    let iframes = document.querySelectorAll('iframe');
    iframes.forEach(iframe => iframe.classList.remove('active'));
    document.getElementById('plot' + (index + 1)).classList.add('active');
}

function showInfoPopup() {
    document.getElementById('popupTitle').innerText = name;
    document.getElementById('popup').style.display = 'block';
    document.getElementById('overlay').style.display = 'block';
    document.getElementById('content').classList.add('blurred');
}

function closeInfoPopup() {
    document.getElementById('popup').style.display = 'none';
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('content').classList.remove('blurred');
}

function backToSelection() {
    window.location.href = '../../index.html';
}

showPlot(0)