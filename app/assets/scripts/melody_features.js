let player = document.getElementById("mplayer");
let vis = document.getElementById("mvis");
let playbackLine = document.getElementById("playback-line")
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
    player.src = `../../assets/midi/melody_features/${pianistFmt}_${conceptPadded}.mid`
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

    document.getElementById("progressionName").innerHTML = "Pattern: " + intervalsToPitches(concept);

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

function intervalsToPitches(intervals) {
    let feature_arr = JSON.parse(intervals)
    let pitch_set = [0]
    let current_pitch = 0
    for (let interval_idx in feature_arr) {
        let current_interval = feature_arr[interval_idx]
        current_pitch = current_pitch + current_interval
        pitch_set.push(current_pitch)
    }
    return "(" + pitch_set.join(", ") + ")"
}

function setTrack(assetPath) {
    let midPath = `../../assets/midi/melody_examples/${assetPath}`
    player.src = midPath
    vis.src = midPath
    playbackLine.style.display = 'block'

    setInterval(() => {
        try {
            var xPos = getCurrentX();
            let rect = document.getElementById("mvis").getBoundingClientRect();
            playbackLine.style.left = Number(xPos) + Number(rect.left) + 'px';
            playbackLine.style.height = (Number(rect.height) - 5) + 'px';
        } catch (err) {
        }
    }, 100);
}

function backToSelection() {
    window.location.href = '../../index.html';
}

function getCurrentX() {
    let svgIndex = null
    for (const [index, element] of vis.noteSequence.notes.entries()) {
        if (element.startTime === player.currentTime) {
            svgIndex = index
        }
    }
    return vis.visualizer.svg.children[svgIndex].getAttribute('x')
}