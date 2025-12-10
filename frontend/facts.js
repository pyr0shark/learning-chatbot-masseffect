// Facts page functionality
class FactsPage {
    constructor() {
        this.approvedContainer = document.getElementById('approved-facts');
        this.rejectedContainer = document.getElementById('rejected-facts');
        this.approvedCount = document.getElementById('approved-count');
        this.rejectedCount = document.getElementById('rejected-count');
        this.resetBtn = document.getElementById('reset-index-btn');
        
        // Load facts when page becomes active
        this.init();
    }
    
    init() {
        // Check if we're on the facts page
        const factsPage = document.getElementById('facts-page');
        if (factsPage && factsPage.classList.contains('active')) {
            this.loadFacts();
        }
        
        // Set up reset button
        if (this.resetBtn) {
            this.resetBtn.addEventListener('click', () => this.resetIndex());
        }
        
        // Listen for page changes via hash changes (router navigation)
        window.addEventListener('hashchange', () => {
            const hash = window.location.hash.slice(1);
            if (hash === 'facts' || hash === '') {
                const factsPage = document.getElementById('facts-page');
                if (factsPage && factsPage.classList.contains('active')) {
                    this.loadFacts();
                }
            }
        });
        
        // Also listen for class changes on the facts page
        if (factsPage) {
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                        if (factsPage.classList.contains('active')) {
                            this.loadFacts();
                        }
                    }
                });
            });
            observer.observe(factsPage, { attributes: true, attributeFilter: ['class'] });
        }
    }
    
    async loadFacts() {
        try {
            const response = await fetch('/api/facts/all');
            if (!response.ok) {
                throw new Error('Failed to load facts');
            }
            
            const data = await response.json();
            this.renderFacts(data.approved, data.rejected);
        } catch (error) {
            console.error('Error loading facts:', error);
            this.approvedContainer.innerHTML = '<div class="error-message">Error loading approved facts</div>';
            this.rejectedContainer.innerHTML = '<div class="error-message">Error loading rejected facts</div>';
        }
    }
    
    renderFacts(approved, rejected) {
        // Update counts
        this.approvedCount.textContent = approved.length;
        this.rejectedCount.textContent = rejected.length;
        
        // Render approved facts
        if (approved.length === 0) {
            this.approvedContainer.innerHTML = '<div class="empty-message">No approved facts yet</div>';
        } else {
            this.approvedContainer.innerHTML = approved.map(fact => this.renderFactCard(fact, 'approved')).join('');
        }
        
        // Render rejected facts
        if (rejected.length === 0) {
            this.rejectedContainer.innerHTML = '<div class="empty-message">No rejected facts yet</div>';
        } else {
            this.rejectedContainer.innerHTML = rejected.map(fact => this.renderFactCard(fact, 'rejected')).join('');
        }
        
        // Attach delete button listeners
        this.attachDeleteListeners();
    }
    
    renderFactCard(fact, status) {
        // Format timestamps
        let dateStr = 'Unknown date';
        let dateLabel = '';
        
        if (status === 'approved' && fact.approved_at) {
            dateStr = new Date(fact.approved_at).toLocaleString();
            dateLabel = 'Approved: ';
        } else if (status === 'rejected' && fact.rejected_at) {
            dateStr = new Date(fact.rejected_at).toLocaleString();
            dateLabel = 'Rejected: ';
        } else if (fact.created_at) {
            dateStr = new Date(fact.created_at).toLocaleString();
            dateLabel = 'Created: ';
        }
        
        // Show source info (optional, for debugging)
        const sourceInfo = fact.source === 'index' ? ' (from index)' : '';
        
        // Delete button (only for approved facts that are user_approved, not database.txt)
        // Only show delete button if source is 'index' or 'session' (user-approved facts)
        const escapedFact = this.escapeAttribute(fact.fact);
        const deleteButton = (status === 'approved' && fact.source !== 'database.txt')
            ? `<button class="fact-delete-btn" data-fact-text="${escapedFact}" title="Delete from index">✕</button>`
            : '';
        
        return `
            <div class="fact-card ${status}" data-fact-text="${escapedFact}">
                <div class="fact-card-header">
                    <div class="fact-header-left">
                        <span class="fact-status-badge ${status}">${status === 'approved' ? '✅ Approved' : '❌ Rejected'}</span>
                        <span class="fact-date">${dateLabel}${dateStr}${sourceInfo}</span>
                    </div>
                    ${deleteButton}
                </div>
                <div class="fact-card-content">
                    <p>${this.escapeHtml(fact.fact)}</p>
                </div>
            </div>
        `;
    }
    
    attachDeleteListeners() {
        // Attach event listeners to delete buttons
        document.querySelectorAll('.fact-delete-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const factText = e.target.getAttribute('data-fact-text');
                this.deleteFact(factText);
            });
        });
    }
    
    async deleteFact(factText) {
        if (!confirm('Are you sure you want to delete this fact from the index? This action cannot be undone.')) {
            return;
        }
        
        try {
            const response = await fetch('/api/facts/delete', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    fact_text: factText
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to delete fact');
            }
            
            // Remove the fact card from the UI
            const factCard = document.querySelector(`[data-fact-text="${this.escapeHtml(factText)}"]`);
            if (factCard) {
                factCard.style.opacity = '0';
                factCard.style.transform = 'translateX(-20px)';
                setTimeout(() => {
                    factCard.remove();
                    // Reload facts to update counts
                    this.loadFacts();
                }, 300);
            } else {
                // If card not found, just reload
                this.loadFacts();
            }
        } catch (error) {
            console.error('Error deleting fact:', error);
            alert('Error deleting fact: ' + error.message);
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    escapeAttribute(text) {
        // Escape for HTML attributes (quotes, etc.)
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }
    
    async resetIndex() {
        if (!confirm('Are you sure you want to reset the index? This will remove all user-approved facts from the index, keeping only the original database.txt content. This action cannot be undone.')) {
            return;
        }
        
        if (this.resetBtn) {
            this.resetBtn.disabled = true;
            this.resetBtn.style.opacity = '0.6';
        }
        
        try {
            const response = await fetch('/api/index/reset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to reset index');
            }
            
            const data = await response.json();
            
            // Show success message
            alert(`Index reset successfully!\n\nRemoved ${data.chunks_removed} user-approved chunks.\nRemaining chunks: ${data.remaining_chunks}`);
            
            // Reload facts to reflect the changes
            this.loadFacts();
            
        } catch (error) {
            console.error('Error resetting index:', error);
            alert('Error resetting index: ' + error.message);
        } finally {
            if (this.resetBtn) {
                this.resetBtn.disabled = false;
                this.resetBtn.style.opacity = '1';
            }
        }
    }
}

// Initialize facts page when DOM is loaded
let factsPage;
document.addEventListener('DOMContentLoaded', () => {
    factsPage = new FactsPage();
});

