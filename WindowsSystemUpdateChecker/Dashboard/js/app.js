// System Maintenance Dashboard - JavaScript

// ============================================
// STATE
// ============================================

let windowsUpdates = [];
let appUpdates = [];
let cleanableItems = [];

// ============================================
// UTILITIES
// ============================================

// Injected into index.html by the server when the page is served.
const DASHBOARD_TOKEN = document.querySelector('meta[name="dashboard-token"]').content;

// All API traffic goes through here so the session token cannot be forgotten on
// a new call site. The server rejects any /api request without it.
function apiFetch(url, options = {}) {
    const headers = Object.assign({}, options.headers, {
        'X-Dashboard-Token': DASHBOARD_TOKEN
    });
    return fetch(url, Object.assign({}, options, { headers }));
}

function apiPost(url, payload) {
    return apiFetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
}

// Package names and update titles come from winget manifests and Windows
// Update, i.e. from outside this machine. They were being interpolated straight
// into innerHTML, so a crafted name could inject markup into a page that can
// launch elevated processes.
function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');

    toast.className = 'toast ' + type;
    toastMessage.textContent = message;

    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

// ============================================
// TAB NAVIGATION
// ============================================

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tabName) {
            btn.classList.add('active');
        }
    });

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById('tab-' + tabName).classList.add('active');

    // Render data for the tab (data is already loaded)
    if (tabName === 'updates') {
        renderWindowsUpdates();
        renderAppUpdates();
    } else if (tabName === 'cleanable') {
        renderCleanableItems();
    }
}

// Initialize tab click handlers
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        switchTab(btn.dataset.tab);
    });
});

// ============================================
// API CALLS
// ============================================

async function fetchStatus() {
    try {
        const response = await apiFetch('/api/status');
        return await response.json();
    } catch (error) {
        console.error('Error fetching status:', error);
        return null;
    }
}

async function loadWindowsUpdates() {
    try {
        const response = await apiFetch('/api/updates/windows');
        const data = await response.json();
        windowsUpdates = data.items || [];
        renderWindowsUpdates();
    } catch (error) {
        console.error('Error loading Windows updates:', error);
        document.getElementById('windowsUpdatesList').innerHTML =
            '<div class="empty-state">Failed to load Windows updates</div>';
    }
}

async function loadAppUpdates() {
    try {
        const response = await apiFetch('/api/updates/apps');
        const data = await response.json();
        appUpdates = data.items || [];
        renderAppUpdates();
    } catch (error) {
        console.error('Error loading app updates:', error);
        document.getElementById('appUpdatesList').innerHTML =
            '<div class="empty-state">Failed to load app updates</div>';
    }
}

async function loadCleanableItems() {
    try {
        const response = await apiFetch('/api/cleanable');
        const data = await response.json();
        cleanableItems = data.items || [];
        renderCleanableItems();
    } catch (error) {
        console.error('Error loading cleanable items:', error);
        document.getElementById('cleanableList').innerHTML =
            '<div class="empty-state">Failed to load cleanable items</div>';
    }
}

// ============================================
// RENDER FUNCTIONS
// ============================================

function renderWindowsUpdates() {
    const container = document.getElementById('windowsUpdatesList');

    if (windowsUpdates.length === 0) {
        container.innerHTML = '<div class="empty-state">No Windows updates available - you\'re up to date!</div>';
        return;
    }

    let html = '';
    windowsUpdates.forEach((update, index) => {
        const severityClass = update.severityClass || 'optional';
        const severityBadge = update.severity === 'CRITICAL'
            ? '<span class="severity-badge critical">CRITICAL</span>'
            : update.severity === 'Important'
                ? '<span class="severity-badge important">Important</span>'
                : '<span class="severity-badge optional">Optional</span>';

        const typeBadge = update.type === 'Driver'
            ? '<span class="type-badge driver">Driver</span>'
            : '<span class="type-badge windows">Windows</span>';

        html += `
            <div class="item-row ${severityClass}">
                <label class="item-checkbox">
                    <input type="checkbox" data-type="windows" data-id="${escapeHtml(update.id)}" data-index="${index}"
                           onchange="updateWindowsSelection()" ${severityClass === 'critical' ? 'checked' : ''}>
                    <span class="checkmark"></span>
                </label>
                <div class="item-content">
                    <div class="item-title">
                        ${typeBadge} ${severityBadge}
                        ${escapeHtml(update.title)}
                    </div>
                    <div class="item-meta">
                        ${escapeHtml(update.sizeFormatted || '')} ${update.kbArticles && update.kbArticles.length ? '| KB' + escapeHtml(update.kbArticles.join(', KB')) : ''}
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
    updateWindowsSelection();
}

function renderAppUpdates() {
    const container = document.getElementById('appUpdatesList');

    if (appUpdates.length === 0) {
        container.innerHTML = '<div class="empty-state">No app updates available - all apps are up to date!</div>';
        return;
    }

    // Group by category
    const grouped = {};
    appUpdates.forEach((app, index) => {
        const cat = app.category || 'Other';
        if (!grouped[cat]) grouped[cat] = [];
        grouped[cat].push({ ...app, index });
    });

    // Sort categories: Security first, then Development, then Other
    const categoryOrder = ['Security/Browser', 'Development', 'Other'];
    const sortedCategories = Object.keys(grouped).sort((a, b) => {
        return categoryOrder.indexOf(a) - categoryOrder.indexOf(b);
    });

    let html = '';
    sortedCategories.forEach(category => {
        const priorityClass = category === 'Security/Browser' ? 'high' : category === 'Development' ? 'medium' : 'low';
        const categoryIcon = category === 'Security/Browser' ? '&#128274;' : category === 'Development' ? '&#128187;' : '&#128230;';

        html += `<div class="category-group">
            <div class="category-header ${priorityClass}">
                <span class="category-icon">${categoryIcon}</span>
                ${escapeHtml(category)}
                <span class="category-count">${grouped[category].length}</span>
                ${category === 'Security/Browser' ? '<span class="recommended-badge">RECOMMENDED</span>' : ''}
            </div>`;

        grouped[category].forEach(app => {
            html += `
                <div class="item-row ${priorityClass}">
                    <label class="item-checkbox">
                        <input type="checkbox" data-type="app" data-id="${escapeHtml(app.id)}" data-index="${app.index}"
                               data-category="${escapeHtml(category)}" onchange="updateAppSelection()"
                               ${category === 'Security/Browser' ? 'checked' : ''}>
                        <span class="checkmark"></span>
                    </label>
                    <div class="item-content">
                        <div class="item-title">${escapeHtml(app.name)}</div>
                        <div class="item-meta">
                            ${escapeHtml(app.currentVersion)} &rarr; <strong>${escapeHtml(app.availableVersion)}</strong>
                        </div>
                    </div>
                </div>
            `;
        });

        html += '</div>';
    });

    container.innerHTML = html;
    updateAppSelection();
}

function renderCleanableItems() {
    const container = document.getElementById('cleanableList');

    if (cleanableItems.length === 0) {
        container.innerHTML = '<div class="empty-state">No cleanable items found - your system is clean!</div>';
        return;
    }

    // Group by category
    const grouped = {};
    cleanableItems.forEach((item, index) => {
        const cat = item.category || 'Other';
        if (!grouped[cat]) grouped[cat] = [];
        grouped[cat].push({ ...item, index });
    });

    const categoryOrder = ['System', 'Browser', 'AI/ML'];
    const sortedCategories = Object.keys(grouped).sort((a, b) => {
        const aIdx = categoryOrder.indexOf(a);
        const bIdx = categoryOrder.indexOf(b);
        return (aIdx === -1 ? 999 : aIdx) - (bIdx === -1 ? 999 : bIdx);
    });

    let html = '';
    sortedCategories.forEach(category => {
        const categoryIcon = category === 'System' ? '&#128187;' : category === 'Browser' ? '&#127760;' : '&#129302;';
        const categoryTotal = grouped[category].reduce((sum, item) => sum + (item.size || 0), 0);

        html += `<div class="category-group">
            <div class="category-header">
                <span class="category-icon">${categoryIcon}</span>
                ${escapeHtml(category)}
                <span class="category-size">${formatBytes(categoryTotal)}</span>
            </div>`;

        grouped[category].forEach(item => {
            const riskClass = item.risk || 'safe';
            const riskBadge = riskClass === 'review'
                ? '<span class="risk-badge review">Review</span>'
                : '<span class="risk-badge safe">Safe</span>';

            html += `
                <div class="item-row">
                    <label class="item-checkbox">
                        <input type="checkbox" data-type="clean" data-id="${escapeHtml(item.id)}" data-index="${item.index}"
                               data-size="${item.size || 0}" data-risk="${escapeHtml(item.risk)}"
                               onchange="updateCleanSelection()" ${riskClass === 'safe' ? 'checked' : ''}>
                        <span class="checkmark"></span>
                    </label>
                    <div class="item-content">
                        <div class="item-title">
                            ${escapeHtml(item.name)}
                            ${riskBadge}
                        </div>
                        <div class="item-meta">
                            <strong>${escapeHtml(item.sizeFormatted || formatBytes(item.size))}</strong>
                            <span class="item-desc">${escapeHtml(item.description || '')}</span>
                        </div>
                    </div>
                </div>
            `;
        });

        html += '</div>';
    });

    container.innerHTML = html;
    updateCleanSelection();
}

// ============================================
// SELECTION FUNCTIONS
// ============================================

function updateWindowsSelection() {
    const checkboxes = document.querySelectorAll('input[data-type="windows"]:checked');
    const btn = document.getElementById('installWindowsBtn');
    btn.disabled = checkboxes.length === 0;
    btn.textContent = checkboxes.length > 0 ? `Install Selected (${checkboxes.length})` : 'Install Selected';
}

function updateAppSelection() {
    const checkboxes = document.querySelectorAll('input[data-type="app"]:checked');
    const btn = document.getElementById('installAppsBtn');
    const countSpan = document.getElementById('selectedAppsCount');

    btn.disabled = checkboxes.length === 0;
    countSpan.textContent = checkboxes.length;
}

function updateCleanSelection() {
    const checkboxes = document.querySelectorAll('input[data-type="clean"]:checked');
    const btn = document.getElementById('cleanBtn');
    const sizeSpan = document.getElementById('selectedCleanSize');

    let totalSize = 0;
    checkboxes.forEach(cb => {
        totalSize += parseInt(cb.dataset.size || 0);
    });

    btn.disabled = checkboxes.length === 0;
    sizeSpan.textContent = formatBytes(totalSize);
}

function selectAllWindows() {
    document.querySelectorAll('input[data-type="windows"]').forEach(cb => cb.checked = true);
    updateWindowsSelection();
}

function selectAllApps() {
    document.querySelectorAll('input[data-type="app"]').forEach(cb => cb.checked = true);
    updateAppSelection();
}

function selectSecurityApps() {
    document.querySelectorAll('input[data-type="app"]').forEach(cb => {
        cb.checked = cb.dataset.category === 'Security/Browser';
    });
    updateAppSelection();
}

function selectAllCleanable() {
    document.querySelectorAll('input[data-type="clean"]').forEach(cb => cb.checked = true);
    updateCleanSelection();
}

function selectAllSafe() {
    document.querySelectorAll('input[data-type="clean"]').forEach(cb => {
        cb.checked = cb.dataset.risk === 'safe';
    });
    updateCleanSelection();
}

// ============================================
// ACTION FUNCTIONS
// ============================================

async function installSelectedWindows() {
    showToast('Launching Windows Update installer...', 'info');

    try {
        const response = await apiPost('/api/action/update', { type: 'windows' });
        const result = await response.json();

        if (result.success) {
            showToast(result.message, 'success');
        } else {
            showToast('Error: ' + result.message, 'error');
        }
    } catch (error) {
        showToast('Error: ' + error.message, 'error');
    }
}

async function installSelectedApps() {
    const checkboxes = document.querySelectorAll('input[data-type="app"]:checked');
    const ids = Array.from(checkboxes).map(cb => cb.dataset.id);

    if (ids.length === 0) {
        showToast('No apps selected', 'warning');
        return;
    }

    showToast(`Installing ${ids.length} app(s)...`, 'info');

    try {
        const response = await apiPost('/api/action/update', { type: 'apps', ids: ids });
        const result = await response.json();

        if (result.success) {
            showToast(result.message, 'success');
        } else {
            showToast('Error: ' + result.message, 'error');
        }
    } catch (error) {
        showToast('Error: ' + error.message, 'error');
    }
}

async function cleanSelected() {
    const checkboxes = document.querySelectorAll('input[data-type="clean"]:checked');
    const ids = Array.from(checkboxes).map(cb => cb.dataset.id);

    if (ids.length === 0) {
        showToast('No items selected', 'warning');
        return;
    }

    showToast('Launching System Cleaner...', 'info');

    try {
        const response = await apiPost('/api/action/clean', { ids: ids });
        const result = await response.json();

        if (result.success) {
            showToast(result.message, 'success');
        } else {
            showToast('Error: ' + result.message, 'error');
        }
    } catch (error) {
        showToast('Error: ' + error.message, 'error');
    }
}

async function launchTool(tool) {
    showToast('Launching tool...', 'info');

    const endpoints = {
        'updater': { type: 'apps', ids: [] },
        'windows': { type: 'windows' },
        'cleaner': { type: 'clean', ids: [] },
        'ai-cleaner': { type: 'clean', ids: [] },
        'drivers': { type: 'drivers' },
        'startup': { type: 'startup' },
        'report': { type: 'report' },
        'full': { type: 'full' }
    };

    // For tools that have specific scripts, we'll use the action endpoint
    if (tool === 'updater') {
        const response = await apiPost('/api/action/update', { type: 'apps' });
    } else if (tool === 'windows') {
        const response = await apiPost('/api/action/update', { type: 'windows' });
    } else if (tool === 'cleaner' || tool === 'ai-cleaner') {
        const response = await apiPost('/api/action/clean', { ids: [] });
    } else {
        showToast('Tool will open in PowerShell window', 'info');
    }
}

function launchFullMaintenance() {
    showToast('Opening Full Maintenance in PowerShell...', 'info');
    launchTool('full');
}

// ============================================
// UI UPDATES
// ============================================

function updateDashboard(status) {
    if (!status) return;

    // Header info
    document.getElementById('computerName').textContent = status.computerName || 'Unknown';
    document.getElementById('lastUpdated').textContent = 'Updated: ' + new Date().toLocaleTimeString();

    // Summary counts
    const windowsCount = status.summary?.windowsUpdates || 0;
    const appCount = status.summary?.appUpdates || 0;
    const criticalCount = status.summary?.criticalUpdates || 0;
    const cleanableFormatted = status.summary?.totalCleanableFormatted || '0 B';

    document.getElementById('windowsUpdatesCount').textContent = windowsCount;
    document.getElementById('appUpdatesCount').textContent = appCount;
    document.getElementById('totalCleanable').textContent = cleanableFormatted;

    // Critical badge
    const criticalBadge = document.getElementById('criticalBadge');
    if (criticalCount > 0) {
        criticalBadge.classList.remove('hidden');
        document.getElementById('criticalCount').textContent = criticalCount;
    } else {
        criticalBadge.classList.add('hidden');
    }

    // Tab badges
    const totalUpdates = windowsCount + appCount;
    const updatesBadge = document.getElementById('updatesTabBadge');
    if (totalUpdates > 0) {
        updatesBadge.textContent = totalUpdates;
        updatesBadge.classList.remove('hidden');
    } else {
        updatesBadge.classList.add('hidden');
    }

    // Disk
    const diskPercent = status.disk?.percentFree || 0;
    document.getElementById('diskPercent').textContent = diskPercent + '%';

    const diskProgress = document.getElementById('diskProgress');
    const usedPercent = 100 - diskPercent;
    diskProgress.style.width = usedPercent + '%';

    if (diskPercent < 10) {
        diskProgress.className = 'progress-fill danger';
        document.getElementById('diskPercent').className = 'stat-value danger';
    } else if (diskPercent < 20) {
        diskProgress.className = 'progress-fill warning';
        document.getElementById('diskPercent').className = 'stat-value warning';
    } else {
        diskProgress.className = 'progress-fill';
        document.getElementById('diskPercent').className = 'stat-value success';
    }

    // Status banner
    updateStatusBanner(status);
}

function updateStatusBanner(status) {
    const banner = document.getElementById('statusBanner');
    const statusText = document.getElementById('statusText');
    const criticalCount = status.summary?.criticalUpdates || 0;
    const totalUpdates = (status.summary?.windowsUpdates || 0) + (status.summary?.appUpdates || 0);
    const cleanable = status.summary?.totalCleanable || 0;

    if (criticalCount > 0) {
        banner.className = 'status-banner status-critical';
        statusText.textContent = `${criticalCount} Critical Update(s) Need Attention!`;
        document.querySelector('.status-icon').innerHTML = '&#9888;';
    } else if (totalUpdates > 5 || cleanable > 5 * 1024 * 1024 * 1024) {
        banner.className = 'status-banner status-warning';
        statusText.textContent = 'Maintenance Recommended';
        document.querySelector('.status-icon').innerHTML = '&#9888;';
    } else {
        banner.className = 'status-banner status-good';
        statusText.textContent = 'System Status: Good';
        document.querySelector('.status-icon').innerHTML = '&#9989;';
    }
}

// ============================================
// INITIALIZATION
// ============================================

async function loadAllData() {
    // Fetch all detailed data first
    try {
        const [winResponse, appResponse, cleanResponse, statusResponse] = await Promise.all([
            apiFetch('/api/updates/windows'),
            apiFetch('/api/updates/apps'),
            apiFetch('/api/cleanable'),
            apiFetch('/api/status')
        ]);

        const winData = await winResponse.json();
        const appData = await appResponse.json();
        const cleanData = await cleanResponse.json();
        const status = await statusResponse.json();

        // Store detailed data
        windowsUpdates = winData.items || [];
        appUpdates = appData.items || [];
        cleanableItems = cleanData.items || [];

        // Override summary counts with actual data counts for consistency
        if (status.summary) {
            status.summary.windowsUpdates = windowsUpdates.length;
            status.summary.appUpdates = appUpdates.length;
            status.summary.criticalUpdates = windowsUpdates.filter(u => u.severityClass === 'critical').length;
        }

        // Update the dashboard with consistent counts
        updateDashboard(status);

        // Render details if on those tabs
        const activeTab = document.querySelector('.tab-btn.active');
        if (activeTab) {
            const tabName = activeTab.dataset.tab;
            if (tabName === 'updates') {
                renderWindowsUpdates();
                renderAppUpdates();
            } else if (tabName === 'cleanable') {
                renderCleanableItems();
            }
        }

        return status;
    } catch (error) {
        console.error('Error loading data:', error);
        return null;
    }
}

async function refreshAll() {
    showToast('Refreshing...', 'info');
    await loadAllData();
    showToast('Refreshed!', 'success');
}

document.addEventListener('DOMContentLoaded', async () => {
    await loadAllData();

    // Auto-refresh every 2 minutes
    setInterval(async () => {
        await loadAllData();
    }, 120000);
});
