let player = document.getElementById("mplayer");
let vis = document.getElementById("mvis");
player.addEventListener("load", _playHandler);


function _playHandler() {
    player.stop();
    player.currentTime = 0.0;
    player.start();
}

function formatPianistName(pianist) {
    return pianist.replace(/ /g, "_").toLowerCase()
}

function parseJSON(jsName) {
    let request = new XMLHttpRequest();
    request.open("GET", jsName, false);
    request.send(null)
    return JSON.parse(request.responseText);
}

function playExampleMidi(concept, pianist) {
    let conceptPadded = String(concept).padStart(3, '0')
    let pianistFmt = formatPianistName(pianist)
    let midPath = `../../assets/midi/melody_features/${pianistFmt}_${conceptPadded}.mid`
    player.src = midPath
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

function popSelect(concept, pianist) {
    let pianistFmt = formatPianistName(pianist)
    let jsPath = `../../assets/metadata/melody_features/${pianistFmt}.json`
    let jsonData = parseJSON(jsPath);
    let conceptData = jsonData[concept]
    let dropper = document.getElementById("dropdown-menu")
    dropper.innerHTML = "";

    document.getElementById("progressionName").innerHTML = concept;

    for (var i in conceptData) {
        let trackData = conceptData[i];
        let opt = document.createElement('option');
        opt.value = trackData["asset_path"] + '.mid';
        opt.innerHTML = `${trackData["track_name"]} (${trackData["album_name"]}, ${trackData["recording_year"]})`
        dropper.appendChild(opt);
    }
    dropper.value = conceptData[0]["asset_path"] + '.mid'
    dropper.onchange(dropper.value)
}

function setTrack(assetPath) {
    let midPath = `../../assets/midi/melody_examples/${assetPath}`
    player.src = midPath
    vis.src = midPath
}

function backToSelection() {
    window.location.href = '../../index.html';
}
