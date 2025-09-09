console.log("aaaaa")

const container = document.getElementById("results");
const form = document.getElementById("prompt-form");

const followupContainer = document.getElementById("follow-up");
const addPromptBtn = document.getElementById("add-prompt-btn");
let promptCount = 1; // start at 1

addPromptBtn.addEventListener("click", () => {
    promptCount++;

    // wrapper for prompt + remove button
    const wrapper = document.createElement("div");
    wrapper.className = "prompt-wrapper";

    const label = document.createElement("label");
    label.setAttribute("for", `claude-p${promptCount}`);
    label.textContent = `Follow Up Prompt ${promptCount}:`;

    const textarea = document.createElement("textarea");
    textarea.id = `claude-p${promptCount}`;
    textarea.name = "claude-prompts"; // consistent name for all
    textarea.rows = 4;
    textarea.cols = 50;

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.textContent = "- remove";
    removeBtn.style.marginLeft = "10px";
    removeBtn.addEventListener("click", () => {
        followupContainer.removeChild(wrapper);
    });

    wrapper.appendChild(label);
    wrapper.appendChild(textarea);
    wrapper.appendChild(removeBtn);

    followupContainer.appendChild(wrapper);
});

form.addEventListener("submit", function(e) {
    e.preventDefault(); // prevent page reload
    // container.innerHTML = ""; // clear previous responses

    const mode = form.elements["mode"].value;
    let promptData;

    if (mode === "single") {
        // Single prompt mode: just take the textarea
        promptData = {
            mode: "single",
            model: form.elements["model"].value,
            initial: form.elements["initial-prompt"].value,
            prompt: form.elements["single-prompt"].value
        };
    } else {
        // Multiple prompt mode: gather all inputs
        const followups = Array.from(form.querySelectorAll("textarea[name='claude-prompts']"))
            .map(el => el.value)
            .filter(v => v.trim() !== ""); // skip empty

        promptData = {
            mode: "multiple",
            model: form.elements["model"].value,
            initial: form.elements["initial-prompt"].value,
            prompts: followups
        };
    }

    // Open SSE stream with JSON payload encoded in query string
    const evtSource = new EventSource(
        `/stream?data=${encodeURIComponent(JSON.stringify(promptData))}`
    );

    evtSource.onmessage = function(event) {
        const responses = JSON.parse(event.data);
        let form_section = document.getElementById('db-form-inner');
        form_section.innerHTML = ""; // replace previous content

        const initial_res = responses.shift();
        const initialDiv = document.getElementById("initial");
        initialDiv.innerText = `${initial_res}`;

        responses.forEach((r, i) => {
            // Create container div for each response
            const responseDiv = document.createElement("div");
            responseDiv.className = "response-item";
            responseDiv.style.marginBottom = "15px";
            
            // Create label
            const label = document.createElement("label");
            const spk_num = i % 2 == 0 ? "1" : "2";
            label.innerHTML = `<b>Speaker ${spk_num}:</b>`;
            label.style.display = "block";
            label.style.marginBottom = "5px";
            
            // Create textarea
            const textarea = document.createElement("textarea");
            textarea.name = `response_${i}`;
            textarea.value = r;
            // Store speaker info as data attribute
            textarea.dataset.speaker = spk_num;
            
            responseDiv.appendChild(label);
            responseDiv.appendChild(textarea);
            form_section.appendChild(responseDiv);
        });
    };

    evtSource.onerror = function() {
        evtSource.close(); // close stream on error
    };
});

const db_form = document.getElementById('db-form');
db_form.addEventListener("submit", function(e) {
    e.preventDefault(); // prevent page reload
    const initial = document.getElementById("initial").innerText
    const dialogueData = {
        prompt: form.elements["initial-prompt"].value,
        initial_response: initial,
    };

    const textareas = db_form.querySelectorAll('textarea');
    
    // Build blocks array from textarea values
    const blocks = [];
    console.log("textareas", textareas)

    textareas.forEach((textarea) => {
        blocks.push({
            speaker: textarea.dataset.speaker,
            text: textarea.value.trim()
        });
    });

    dialogueData.blocks = blocks;
    // Call the function
    sendDataToFlask(dialogueData)
        .then(result => {
            console.log('Dialogue inserted successfully:', result);
        })
        .catch(error => {
            console.error('Failed to insert dialogue:', error);
        });
})

// JavaScript to send data to Flask endpoint using URL parameters
async function sendDataToFlask(data) {
    try {
        // Convert data object to JSON string for the URL parameter
        const dataParam = encodeURIComponent(JSON.stringify(data));
        const url = `/insert?data=${dataParam}`;
        
        const response = await fetch(url, {
            method: 'GET', // Using GET since you're reading from request.args
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.text();
        console.log('Success:', result);
        return result;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}


const modeSelect = document.getElementById("mode");
const singleSection = document.getElementById("single-prompt-section");
const multipleSection = document.getElementById("multiple-prompts-section");

function toggleSections() {
    if (modeSelect.value === "single") {
        singleSection.style.display = "block";
        multipleSection.style.display = "none";
    } else {
        singleSection.style.display = "none";
        multipleSection.style.display = "block";
    }
}

// Run on page load (in case default selection is not "multiple")
toggleSections();

// Listen for changes
modeSelect.addEventListener("change", toggleSections);