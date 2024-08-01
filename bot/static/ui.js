
function initErrorPopup() {
    document.getElementById('close-error-popup').addEventListener('click', () => {
        document.getElementById('error-popup').classList.add('hidden');
    });
}

function initPopupMenu(onLoadCallback) {
    const popupButton = document.getElementById('popup-button');
    const popupMenu = document.getElementById('popup-menu');
    document.addEventListener('DOMContentLoaded', function () {
        onLoadCallback();
    });
    popupButton.addEventListener('mouseenter', () => {
        popupMenu.style.display = 'block';
    });
    popupButton.addEventListener('mouseleave', () => {
        popupMenu.style.display = 'none';
    });
    popupMenu.addEventListener('mouseenter', () => {
        popupMenu.style.display = 'block';
    });
    popupMenu.addEventListener('mouseleave', () => {
        popupMenu.style.display = 'none';
    });
}

function initSendButtonResize() {
    window.addEventListener('load', updateSendButton);
    window.addEventListener('resize', updateSendButton);
}

function initTextArea(submittedTextList, sendMessageFunc) {
    let submittedTextSelectionCursor = -1;
    let textSubmissionCandidate = '';
    const textarea = document.getElementById('input');
    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
        if (this.style.height < this.style.maxHeight) {
            this.style.overflow = 'hidden';
        } else {
            this.style.overflow = 'auto';
        }
        // Update with each keystroke
        textSubmissionCandidate = textarea.value;
    });
    textarea.addEventListener('keydown', function(event) {
        if (event.key === 'ArrowUp' && textarea.selectionStart === 0) {
            if (submittedTextList.length > 0 && (submittedTextList.length - 1) > submittedTextSelectionCursor) {
                submittedTextSelectionCursor += 1;
                let reversedSubmissions = submittedTextList.slice().reverse();
                textarea.value = reversedSubmissions[submittedTextSelectionCursor];
            }
            event.preventDefault();
        }
        if (event.key === 'ArrowDown' && textarea.selectionStart === textarea.value.length) {
            if (submittedTextList.length > 0 && submittedTextSelectionCursor > -1) {
                submittedTextSelectionCursor -= 1;
                if (submittedTextSelectionCursor === -1) {
                    textarea.value = textSubmissionCandidate;
                } else {
                    let reversedSubmissions = submittedTextList.slice().reverse();
                    textarea.value = reversedSubmissions[submittedTextSelectionCursor];
                }
            }
            event.preventDefault();
        }
        if (event.key === 'Enter' && (!event.shiftKey)) {
            event.preventDefault();
            sendMessageFunc();

            this.style.height = 'auto';
            this.style.overflow = 'auto';
        }
    });
    window.addEventListener('keydown', (event) => {
        if (document.activeElement !== textarea) {
            textarea.focus()
        }
    })
}

function getAssistantMessageBox(name, content, stopReason = null, iconClass = "fa fa-virus") {
    const assistantMessageBox = document.createElement('div');
    assistantMessageBox.className = 'assistant-message-box';
    assistantMessageBox.innerHTML = `<i class="chat-metadata ${iconClass}"></i><b class="chat-metadata">${name}:</b>`;
    assistantMessageBox.innerHTML += `${marked.parse(content)}</br>`;
    if (stopReason !== null && stopReason !== '') {
        assistantMessageBox.innerHTML += `<i class="chat-metadata">${new Date().toLocaleTimeString()} - ${stopReason}</i> | `;
    }
    return assistantMessageBox;
}

function getCopyTextLink(text) {
    const textLink = document.createElement('a');
    textLink.title = "Copy";
    textLink.className = "chatbox-message-link";
    textLink.innerHTML = `<i class="fa fa-copy">`;
    textLink.addEventListener('click', (event) => {
        event.preventDefault();
        navigator.clipboard.writeText(text)
            .then(() => {
                console.log('Copied text: ' + text);
            })
            .catch(error => {
                console.log('Failed to copy text:', error)
            });
    });
    return textLink;
}

function getLoadingIndicator() {
    const loadingIndicator = document.createElement('div');
    loadingIndicator.className = 'loading-indicator';
    loadingIndicator.innerHTML = '<div class="loading-spinner"></div>';
    return loadingIndicator;
}

function getQueryParams() {
    const params = {};
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);

    urlParams.forEach((value, key) => {
        params[key] = value;
    });
    return params;
}

function getUserMessageBox(message, codeBlockType = "") {
    const userMessageBox = document.createElement('div');
    userMessageBox.className = 'user-message-box';
    userMessageBox.innerHTML = `<i class="chat-metadata fa fa-user"></i><b class="chat-metadata">user:</b>`;
    userMessageBox.innerHTML += `${marked.parse(codeBlockType === '' ? message : `\`\`\`${codeBlockType}\n${message}\n\`\`\``)}</br>`;
    userMessageBox.innerHTML += `<i class="chat-metadata">${new Date().toLocaleTimeString()}</i> | `;
    return userMessageBox;
}

function handleError(error, message) {
    console.error(message, error);
    showErrorPopup(`${message} ${error.message}`);
}

function scrollChatBoxToBottom() {
    const chatbox = document.getElementById('chatbox');
    chatbox.scrollTop = chatbox.scrollHeight;
}

function setVersionMarker(version) {
    const versionMarker = document.getElementById('version-marker');
    versionMarker.innerText = version;
    const chatNavLink = document.getElementById("chat-nav-link")
    chatNavLink.href = `/?bot_version=${version}`
    return version;
}

const showErrorPopup = (message) => {
    const errorPopup = document.getElementById('error-popup');
    const errorMessageText = document.getElementById('error-popup-message-text');

    errorPopup.classList.remove('hidden');
    errorMessageText.innerHTML = message;

    // Auto-hide after 5 seconds
    setTimeout(() => {
        errorPopup.classList.add('hidden');
    }, 5000);
};

function throwOrJson(response) {
    if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return response.json();
}

    // Function to check button width and replace text with icon if needed
function updateSendButton() {
    const sendButton = document.getElementById('send-button');
    if (sendButton.offsetWidth < 60) { // Adjust the width as needed
        sendButton.innerHTML = '<i class="fas fa-arrow-right"></i>';
    } else {
        sendButton.innerHTML = 'Send';
    }
}