/**
 * RAG Chat Frontend Application
 * Handles collection management, PDF uploads, and chat with citations
 */

// ===========================
// State Management
// ===========================
const state = {
    collections: [],
    activeCollection: null,
    documents: [],  // Documents in active collection
    jobs: {},  // job_id -> {filename, status, message}
    messages: [],
    retrievedDocs: [],  // For citation tooltips
    sessionId: generateSessionId()
};

function generateSessionId() {
    return 'session_' + Math.random().toString(36).substring(2, 15);
}

// ===========================
// DOM Elements
// ===========================
const elements = {
    collectionsList: document.getElementById('collectionsList'),
    newCollectionName: document.getElementById('newCollectionName'),
    createCollectionBtn: document.getElementById('createCollectionBtn'),
    uploadSection: document.getElementById('uploadSection'),
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    jobsList: document.getElementById('jobsList'),
    chatHeader: document.getElementById('chatHeader'),
    chatMessages: document.getElementById('chatMessages'),
    chatInputContainer: document.getElementById('chatInputContainer'),
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
    citationTooltip: document.getElementById('citationTooltip'),
    deleteModal: document.getElementById('deleteModal'),
    deleteCollectionName: document.getElementById('deleteCollectionName'),
    cancelDeleteBtn: document.getElementById('cancelDeleteBtn'),
    confirmDeleteBtn: document.getElementById('confirmDeleteBtn'),
    documentsList: document.getElementById('documentsList'),
    deleteDocModal: document.getElementById('deleteDocModal'),
    deleteDocumentName: document.getElementById('deleteDocumentName'),
    cancelDeleteDocBtn: document.getElementById('cancelDeleteDocBtn'),
    confirmDeleteDocBtn: document.getElementById('confirmDeleteDocBtn')
};

// ===========================
// API Functions
// ===========================
const API = {
    async getCollections() {
        const res = await fetch('/collections');
        const data = await res.json();
        return data.collections || [];
    },

    async createCollection(name) {
        const res = await fetch('/collections', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        return res.json();
    },

    async deleteCollection(name) {
        const res = await fetch(`/collections/${name}`, { method: 'DELETE' });
        return res.json();
    },

    async getDocuments(collectionName) {
        const res = await fetch(`/collections/${collectionName}/documents`);
        const data = await res.json();
        return data.documents || [];
    },

    async deleteDocument(collectionName, documentName) {
        const res = await fetch(`/collections/${collectionName}/documents/${encodeURIComponent(documentName)}`, {
            method: 'DELETE'
        });
        return res.json();
    },

    async uploadDocument(collectionName, file) {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch(`/collections/${collectionName}/documents`, {
            method: 'POST',
            body: formData
        });
        return res.json();
    },

    async getJobStatus(jobId) {
        const res = await fetch(`/jobs/${jobId}`);
        return res.json();
    },

    async chat(collectionName, message, sessionId) {
        const res = await fetch(`/collections/${collectionName}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: sessionId })
        });
        return res.json();
    }
};

// ===========================
// Collection Management
// ===========================
async function loadCollections() {
    try {
        state.collections = await API.getCollections();
        renderCollections();
    } catch (err) {
        console.error('Failed to load collections:', err);
    }
}

function renderCollections() {
    if (state.collections.length === 0) {
        elements.collectionsList.innerHTML = `
            <div class="empty-state">
                <div class="icon">📁</div>
                <p>No collections yet.<br>Create one to get started.</p>
            </div>
        `;
        return;
    }

    elements.collectionsList.innerHTML = state.collections.map(name => `
        <div class="collection-item ${state.activeCollection === name ? 'active' : ''}" 
             data-name="${name}">
            <span class="icon">📁</span>
            <span class="name">${name}</span>
            <button class="delete-btn" data-name="${name}">🗑️</button>
        </div>
    `).join('');

    // Add click handlers
    document.querySelectorAll('.collection-item').forEach(item => {
        item.addEventListener('click', (e) => {
            if (!e.target.classList.contains('delete-btn')) {
                selectCollection(item.dataset.name);
            }
        });
    });

    document.querySelectorAll('.delete-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            showDeleteModal(btn.dataset.name);
        });
    });
}

function selectCollection(name) {
    state.activeCollection = name;
    state.messages = [];
    state.retrievedDocs = [];
    state.documents = [];
    state.sessionId = generateSessionId();  // New session for new collection

    renderCollections();
    elements.uploadSection.style.display = 'block';
    elements.chatHeader.innerHTML = `<h2>💬 ${name}</h2>`;
    elements.chatMessages.innerHTML = `
        <div class="empty-state">
            <div class="icon">💬</div>
            <p>Start a conversation with your documents</p>
        </div>
    `;
    elements.chatInputContainer.style.display = 'flex';
    elements.messageInput.focus();

    // Load documents for this collection
    loadDocuments();
}

async function createCollection() {
    const name = elements.newCollectionName.value.trim();
    if (!name) return;

    try {
        await API.createCollection(name);
        elements.newCollectionName.value = '';
        await loadCollections();
        selectCollection(name);
    } catch (err) {
        console.error('Failed to create collection:', err);
    }
}

function showDeleteModal(name) {
    state.collectionToDelete = name;
    elements.deleteCollectionName.textContent = name;
    elements.deleteModal.style.display = 'flex';
}

async function confirmDelete() {
    if (!state.collectionToDelete) return;

    try {
        await API.deleteCollection(state.collectionToDelete);
        elements.deleteModal.style.display = 'none';

        if (state.activeCollection === state.collectionToDelete) {
            state.activeCollection = null;
            elements.uploadSection.style.display = 'none';
            elements.chatHeader.innerHTML = '<h2>💬 Select a collection to start chatting</h2>';
            elements.chatMessages.innerHTML = '';
            elements.chatInputContainer.style.display = 'none';
        }

        await loadCollections();
    } catch (err) {
        console.error('Failed to delete collection:', err);
    }
}

// ===========================
// Document Management
// ===========================
async function loadDocuments() {
    if (!state.activeCollection) return;

    try {
        state.documents = await API.getDocuments(state.activeCollection);
        renderDocuments();
    } catch (err) {
        console.error('Failed to load documents:', err);
    }
}

function renderDocuments() {
    if (state.documents.length === 0) {
        elements.documentsList.innerHTML = `
            <div class="empty-state small">
                <p>No documents yet</p>
            </div>
        `;
        return;
    }

    elements.documentsList.innerHTML = state.documents.map(doc => {
        const pdfUrl = `/data/raw/${encodeURIComponent(state.activeCollection)}/${encodeURIComponent(doc.filename)}`;
        return `
        <div class="document-item" data-filename="${escapeAttr(doc.filename)}">
            <span class="doc-icon">📄</span>
            <a href="${pdfUrl}" target="_blank" class="doc-name" title="${escapeAttr(doc.source)}">${escapeHtml(doc.filename)}</a>
            <button class="delete-doc-btn" data-filename="${escapeAttr(doc.filename)}">🗑️</button>
        </div>
    `}).join('');

    // Add delete handlers
    document.querySelectorAll('.delete-doc-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            showDeleteDocModal(btn.dataset.filename);
        });
    });
}

function showDeleteDocModal(filename) {
    state.documentToDelete = filename;
    elements.deleteDocumentName.textContent = filename;
    elements.deleteDocModal.style.display = 'flex';
}

async function confirmDeleteDoc() {
    if (!state.documentToDelete || !state.activeCollection) return;

    try {
        await API.deleteDocument(state.activeCollection, state.documentToDelete);
        elements.deleteDocModal.style.display = 'none';
        await loadDocuments();
    } catch (err) {
        console.error('Failed to delete document:', err);
    }
}

// ===========================
// Document Upload & Job Polling
// ===========================
async function uploadFile(file) {
    if (!state.activeCollection) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        alert('Only PDF files are supported');
        return;
    }

    try {
        const result = await API.uploadDocument(state.activeCollection, file);
        state.jobs[result.job_id] = {
            filename: file.name,
            status: result.status,
            message: result.message
        };
        renderJobs();
        pollJobStatus(result.job_id);
    } catch (err) {
        console.error('Upload failed:', err);
    }
}

function renderJobs() {
    const jobEntries = Object.entries(state.jobs);
    if (jobEntries.length === 0) {
        elements.jobsList.innerHTML = '';
        return;
    }

    elements.jobsList.innerHTML = jobEntries.map(([id, job]) => {
        let statusIcon = '';
        let statusClass = '';
        switch (job.status) {
            case 'queued':
            case 'processing':
                statusIcon = '<span class="loading-spinner"></span>';
                statusClass = 'status-processing';
                break;
            case 'completed':
                statusIcon = '✅';
                statusClass = 'status-completed';
                break;
            case 'failed':
                statusIcon = '❌';
                statusClass = 'status-failed';
                break;
        }
        return `
            <div class="job-item" data-id="${id}">
                <span class="filename">${job.filename}</span>
                <span class="status ${statusClass}">${statusIcon}</span>
            </div>
        `;
    }).join('');
}

async function pollJobStatus(jobId) {
    const poll = async () => {
        try {
            const status = await API.getJobStatus(jobId);
            state.jobs[jobId].status = status.status;
            state.jobs[jobId].message = status.message;
            renderJobs();

            if (status.status === 'queued' || status.status === 'processing') {
                setTimeout(poll, 2000);  // Poll every 2 seconds
            } else if (status.status === 'completed') {
                // Refresh documents list when ingestion completes
                loadDocuments();
            }
        } catch (err) {
            console.error('Failed to poll job status:', err);
        }
    };
    poll();
}

// ===========================
// Chat
// ===========================
async function sendMessage() {
    const message = elements.messageInput.value.trim();
    if (!message || !state.activeCollection) return;

    // Add user message
    state.messages.push({ role: 'user', content: message });
    elements.messageInput.value = '';
    renderMessages();

    // Show loading
    const loadingId = 'loading-' + Date.now();
    elements.chatMessages.insertAdjacentHTML('beforeend', `
        <div class="message assistant" id="${loadingId}">
            <span class="loading"></span> Thinking...
        </div>
    `);
    scrollToBottom();

    // Track response time
    const startTime = performance.now();

    try {
        elements.sendBtn.disabled = true;
        const response = await API.chat(state.activeCollection, message, state.sessionId);

        // Calculate response time
        const endTime = performance.now();
        const responseTime = ((endTime - startTime) / 1000).toFixed(2);

        // Remove loading
        document.getElementById(loadingId)?.remove();

        // Store retrieved docs for citations
        state.retrievedDocs = response.retrieved_documents || [];

        // Add assistant message with response time
        state.messages.push({
            role: 'assistant',
            content: response.response,
            docs: state.retrievedDocs,
            responseTime: responseTime
        });
        renderMessages();
    } catch (err) {
        console.error('Chat failed:', err);
        document.getElementById(loadingId)?.remove();
        state.messages.push({
            role: 'assistant',
            content: 'Sorry, an error occurred. Please try again.'
        });
        renderMessages();
    } finally {
        elements.sendBtn.disabled = false;
    }
}

function renderMessages() {
    if (state.messages.length === 0) {
        elements.chatMessages.innerHTML = `
            <div class="empty-state">
                <div class="icon">💬</div>
                <p>Start a conversation with your documents</p>
            </div>
        `;
        return;
    }

    elements.chatMessages.innerHTML = state.messages.map((msg, idx) => {
        let content;
        let responseTimeHtml = '';

        if (msg.role === 'assistant') {
            // For assistant: parse markdown first (on raw content), then citations
            content = parseMarkdown(msg.content);
            if (msg.docs) {
                content = parseCitations(content, msg.docs);
            }
            // Add response time if available
            if (msg.responseTime) {
                responseTimeHtml = `<div class="response-time">⏱️ ${msg.responseTime}s</div>`;
            }
        } else {
            // For user messages: escape HTML for safety
            content = escapeHtml(msg.content);
        }

        return `<div class="message ${msg.role}">${content}${responseTimeHtml}</div>`;
    }).join('');

    // Add citation hover handlers
    document.querySelectorAll('.citation').forEach(el => {
        el.addEventListener('mouseenter', showCitationTooltip);
        el.addEventListener('mouseleave', hideCitationTooltip);
    });

    // Render LaTeX math in chat messages
    renderLatex();

    scrollToBottom();
}

function parseCitations(text, docs) {
    return text.replace(/\[(\d+)\]/g, (match, num) => {
        const index = parseInt(num) - 1;
        if (index >= 0 && index < docs.length) {
            const doc = docs[index];
            return `<span class="citation" 
                data-index="${index}"
                data-type="${doc.type || 'text'}"
                data-source="${escapeAttr(doc.source || '')}"
                data-page="${doc.page_number || ''}"
                data-text="${escapeAttr(doc.text || '')}"
                data-image="${escapeAttr(doc.image_path || '')}"
            >${match}</span>`;
        }
        return match;
    });
}

function showCitationTooltip(e) {
    const el = e.target;
    const type = el.dataset.type;
    const source = el.dataset.source;
    const page = el.dataset.page;
    const text = el.dataset.text;
    const imagePath = el.dataset.image;

    const tooltip = elements.citationTooltip;

    // Clear any pending hide timeout
    if (tooltip.hideTimeout) {
        clearTimeout(tooltip.hideTimeout);
        tooltip.hideTimeout = null;
    }

    // Build source URL - serve PDF from data/raw
    const sourceFilename = getFilename(source);
    const sourceUrl = getSourceUrl(source);

    // Show clickable source filename
    const headerEl = tooltip.querySelector('.tooltip-header');
    headerEl.innerHTML = `
        <a href="${sourceUrl}" target="_blank" class="tooltip-source" title="Open PDF">${sourceFilename}</a>
        <span class="tooltip-page">${page ? `p.${page}` : ''}</span>
    `;

    const contentEl = tooltip.querySelector('.tooltip-content');
    if (type === 'Image' || type === 'Table') {
        // Image or Table - clickable image that opens in new tab
        const imageUrl = getImageUrl(imagePath);
        contentEl.innerHTML = imageUrl
            ? `<a href="${imageUrl}" target="_blank" title="Open image in new tab"><img src="${imageUrl}" alt="Citation image" onerror="this.parentElement.innerHTML='<p>Image not available</p>'"></a>`
            : '<p>Image not available</p>';
    } else {
        // Text types (NarrativeText, Title, ListItem, etc.) - show text content
        contentEl.innerHTML = `<p>${escapeHtml(text)}</p>`;
    }

    // Position tooltip
    const rect = el.getBoundingClientRect();
    tooltip.style.left = `${rect.left}px`;
    tooltip.style.top = `${rect.bottom + 8}px`;

    // Adjust if off screen
    tooltip.classList.add('visible');
    const tooltipRect = tooltip.getBoundingClientRect();
    if (tooltipRect.right > window.innerWidth) {
        tooltip.style.left = `${window.innerWidth - tooltipRect.width - 16}px`;
    }
    if (tooltipRect.bottom > window.innerHeight) {
        tooltip.style.top = `${rect.top - tooltipRect.height - 8}px`;
    }
}

function hideCitationTooltip(e) {
    const tooltip = elements.citationTooltip;

    // Delay hiding to allow moving mouse to tooltip
    tooltip.hideTimeout = setTimeout(() => {
        tooltip.classList.remove('visible');
    }, 150);
}

// Keep tooltip visible when hovering over it
function initTooltipHover() {
    elements.citationTooltip.addEventListener('mouseenter', () => {
        if (elements.citationTooltip.hideTimeout) {
            clearTimeout(elements.citationTooltip.hideTimeout);
            elements.citationTooltip.hideTimeout = null;
        }
    });

    elements.citationTooltip.addEventListener('mouseleave', () => {
        elements.citationTooltip.classList.remove('visible');
    });
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// ===========================
// Utilities
// ===========================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeAttr(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\n/g, '&#10;')
        .replace(/\r/g, '&#13;');
}

// Extract filename from full path
function getFilename(path) {
    if (!path) return '';
    // Handle both forward and back slashes
    return path.split(/[\\/]/).pop();
}

// Convert source path to relative URL for serving PDFs
function getSourceUrl(path) {
    if (!path) return '';
    const normalizedPath = path.replace(/\\/g, '/');

    // If it's an absolute path, extract relative part from 'data/raw'
    const match = normalizedPath.match(/data\/raw\/(.+)$/);
    if (match) {
        return `/data/raw/${match[1]}`;
    }

    // If it's already a relative path like "collection/file.pdf" (new logic)
    if (!normalizedPath.includes(':') && normalizedPath.includes('/')) {
        return `/data/raw/${normalizedPath}`;
    }

    // Fallback: try to find it in the active collection first, then root
    const filename = getFilename(path);
    if (state.activeCollection) {
        return `/data/raw/${encodeURIComponent(state.activeCollection)}/${encodeURIComponent(filename)}`;
    }
    return `/data/raw/${encodeURIComponent(filename)}`;
}

// Use marked.js library for markdown parsing
function parseMarkdown(text) {
    return marked.parse(text);
}

function renderLatex() {
    renderMathInElement(elements.chatMessages, {
        delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true }
        ],
        throwOnError: false
    });
}

// Convert absolute image path to relative URL for serving
function getImageUrl(imagePath) {
    if (!imagePath) return '';

    // Normalize path separators
    const normalizedPath = imagePath.replace(/\\/g, '/');

    // If it's already a relative path, use it directly
    if (!normalizedPath.includes(':')) {
        return `/data/processed/${normalizedPath}`;
    }

    // Extract relative path from absolute path
    // Look for 'data/processed/' in the normalized path
    const match = normalizedPath.match(/data\/processed\/(.+)$/);
    if (match) {
        return `/data/processed/${match[1]}`;
    }

    // Fallback: just use the filename
    return `/data/processed/${getFilename(imagePath)}`;
}

// ===========================
// Event Listeners
// ===========================
function initEventListeners() {
    // Create collection
    elements.createCollectionBtn.addEventListener('click', createCollection);
    elements.newCollectionName.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') createCollection();
    });

    // Delete collection modal
    elements.cancelDeleteBtn.addEventListener('click', () => {
        elements.deleteModal.style.display = 'none';
    });
    elements.confirmDeleteBtn.addEventListener('click', confirmDelete);

    // Delete document modal
    elements.cancelDeleteDocBtn.addEventListener('click', () => {
        elements.deleteDocModal.style.display = 'none';
    });
    elements.confirmDeleteDocBtn.addEventListener('click', confirmDeleteDoc);

    // File upload
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) uploadFile(e.target.files[0]);
        e.target.value = '';  // Reset for same file
    });

    // Drag and drop
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('dragover');
    });
    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('dragover');
    });
    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
    });

    // Chat
    elements.sendBtn.addEventListener('click', sendMessage);
    elements.messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    elements.messageInput.addEventListener('input', () => {
        elements.messageInput.style.height = 'auto';
        elements.messageInput.style.height = Math.min(elements.messageInput.scrollHeight, 150) + 'px';
    });
}

// ===========================
// Initialize
// ===========================
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    initTooltipHover();
    loadCollections();
});
