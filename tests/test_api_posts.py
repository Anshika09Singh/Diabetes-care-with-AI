"""
Unit tests for /api/posts endpoint.
Feature: forum-search-filter
"""
import json
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app
from models import db, Post


@pytest.fixture
def client():
    """Create test client with in-memory database."""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.app_context():
        db.create_all()
        with app.test_client() as test_client:
            yield test_client
        db.drop_all()


@pytest.fixture
def sample_posts(client):
    """Add sample posts for testing."""
    test_posts_data = [
        {'content': 'Managing diabetes with diet', 'timestamp': datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)},
        {'content': 'Exercise tips for blood sugar control', 'timestamp': datetime(2025, 1, 14, 9, 0, 0, tzinfo=timezone.utc)},
        {'content': 'My insulin pump experience', 'timestamp': datetime(2025, 1, 13, 8, 0, 0, tzinfo=timezone.utc)},
        {'content': 'Diet and nutrition questions', 'timestamp': datetime(2025, 1, 12, 7, 0, 0, tzinfo=timezone.utc)},
        {'content': 'New to diabetes diagnosis', 'timestamp': datetime(2025, 1, 11, 6, 0, 0, tzinfo=timezone.utc)},
    ]
    
    with app.app_context():
        for post_data in test_posts_data:
            post = Post(content=post_data['content'], timestamp=post_data['timestamp'])
            db.session.add(post)
        db.session.commit()
    
    return test_posts_data


class TestGetPosts:
    """Tests for GET /api/posts endpoint."""
    
    def test_get_empty_posts(self, client):
        """Test getting posts when none exist."""
        response = client.get('/api/posts')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['posts'] == []
        assert data['pagination']['total'] == 0
        assert data['pagination']['page'] == 1
    
    def test_get_all_posts(self, client, sample_posts):
        """Test getting all posts without filters."""
        response = client.get('/api/posts')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['posts']) == 5
        assert data['pagination']['total'] == 5
    
    def test_search_filter(self, client, sample_posts):
        """Test search query parameter."""
        response = client.get('/api/posts?search=diet')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['posts']) == 2  # 'diet' appears in 2 posts
        for post in data['posts']:
            assert 'diet' in post['content'].lower()
    
    def test_search_case_insensitive(self, client, sample_posts):
        """Test that search is case-insensitive."""
        response_lower = client.get('/api/posts?search=diabetes')
        response_upper = client.get('/api/posts?search=DIABETES')
        
        data_lower = json.loads(response_lower.data)
        data_upper = json.loads(response_upper.data)
        
        assert len(data_lower['posts']) == len(data_upper['posts'])
    
    def test_search_no_results(self, client, sample_posts):
        """Test search with no matching results."""
        response = client.get('/api/posts?search=nonexistentterm')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['posts']) == 0
        assert data['pagination']['total'] == 0
    
    def test_date_filter_start(self, client, sample_posts):
        """Test start_date filter."""
        response = client.get('/api/posts?start_date=2025-01-14')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['posts']) == 2  # Jan 14 and Jan 15
    
    def test_date_filter_end(self, client, sample_posts):
        """Test end_date filter."""
        response = client.get('/api/posts?end_date=2025-01-12')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['posts']) == 2  # Jan 11 and Jan 12
    
    def test_date_filter_range(self, client, sample_posts):
        """Test combined start and end date filter."""
        response = client.get('/api/posts?start_date=2025-01-12&end_date=2025-01-14')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['posts']) == 3  # Jan 12, 13, 14
    
    def test_pagination_default(self, client, sample_posts):
        """Test default pagination (10 per page)."""
        response = client.get('/api/posts')
        data = json.loads(response.data)
        assert data['pagination']['per_page'] == 10
        assert data['pagination']['page'] == 1
    
    def test_pagination_custom_per_page(self, client, sample_posts):
        """Test custom per_page parameter."""
        response = client.get('/api/posts?per_page=2')
        data = json.loads(response.data)
        assert len(data['posts']) == 2
        assert data['pagination']['per_page'] == 2
        assert data['pagination']['total_pages'] == 3
    
    def test_pagination_page_navigation(self, client, sample_posts):
        """Test navigating to different pages."""
        response = client.get('/api/posts?per_page=2&page=2')
        data = json.loads(response.data)
        assert data['pagination']['page'] == 2
        assert len(data['posts']) == 2
    
    def test_combined_filters(self, client, sample_posts):
        """Test combining search and date filters."""
        response = client.get('/api/posts?search=diabetes&start_date=2025-01-11')
        assert response.status_code == 200
        data = json.loads(response.data)
        # Should match posts with 'diabetes' from Jan 11 onwards
        for post in data['posts']:
            assert 'diabetes' in post['content'].lower()


class TestGetPostsValidation:
    """Tests for input validation on GET /api/posts."""
    
    def test_invalid_page_negative(self, client):
        """Test that negative page number is corrected to 1."""
        response = client.get('/api/posts?page=-1')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['pagination']['page'] == 1
    
    def test_invalid_page_zero(self, client):
        """Test that page=0 is corrected to 1."""
        response = client.get('/api/posts?page=0')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['pagination']['page'] == 1
    
    def test_invalid_page_non_integer(self, client):
        """Test error for non-integer page."""
        response = client.get('/api/posts?page=abc')
        assert response.status_code == 400
    
    def test_invalid_per_page_zero(self, client):
        """Test that per_page=0 is corrected to default 10."""
        response = client.get('/api/posts?per_page=0')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['pagination']['per_page'] == 10
    
    def test_invalid_per_page_exceeds_max(self, client):
        """Test that per_page > 50 is corrected to default 10."""
        response = client.get('/api/posts?per_page=100')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['pagination']['per_page'] == 10
    
    def test_invalid_date_format(self, client):
        """Test error for invalid date format."""
        response = client.get('/api/posts?start_date=invalid')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Invalid date format' in data['error']
    
    def test_invalid_date_range(self, client):
        """Test error when start_date > end_date."""
        response = client.get('/api/posts?start_date=2025-01-15&end_date=2025-01-10')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'start_date must be before' in data['error']


class TestPostCreation:
    """Tests for POST /api/posts endpoint."""
    
    def test_create_post(self, client):
        """Test creating a new post."""
        response = client.post('/api/posts',
                               data=json.dumps({'content': 'Test post content'}),
                               content_type='application/json')
        assert response.status_code == 201
        data = json.loads(response.data)
        assert data['content'] == 'Test post content'
        assert 'id' in data
        assert 'timestamp' in data
    
    def test_create_post_empty_content(self, client):
        """Test error when creating post with empty content."""
        response = client.post('/api/posts',
                               data=json.dumps({'content': ''}),
                               content_type='application/json')
        assert response.status_code == 400
    
    def test_create_post_whitespace_content(self, client):
        """Test error when creating post with whitespace-only content."""
        response = client.post('/api/posts',
                               data=json.dumps({'content': '   '}),
                               content_type='application/json')
        assert response.status_code == 400
