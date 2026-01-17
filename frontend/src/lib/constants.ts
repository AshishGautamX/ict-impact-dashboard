import { College } from '@/types';

export const COLLEGES: College[] = [
  { id: '1', name: 'Darbhanga Medical College', code: 'DMC', district: 'Darbhanga' },
  { id: '2', name: 'Indira Gandhi Institute of Medical Sciences', code: 'IGIMS', district: 'Patna' },
  { id: '3', name: 'Patna Medical College and Hospital', code: 'PMCH', district: 'Patna' },
  { id: '4', name: 'Nalanda Medical College and Hospital', code: 'NMCH', district: 'Patna' },
  { id: '5', name: 'Jawaharlal Nehru Medical College', code: 'JNMC', district: 'Bhagalpur' },
  { id: '6', name: 'Vardhman Institute of Medical Sciences', code: 'VIMS', district: 'Pawapuri' },
  { id: '7', name: 'Anugrah Narayan Magadh Medical College', code: 'ANMMC', district: 'Gaya' },
  { id: '8', name: 'Sri Krishna Medical College and Hospital', code: 'SKMCH', district: 'Muzaffarpur' },
  { id: '9', name: 'Government Medical College', code: 'GMC', district: 'Bettiah' },
];

export const AUTOMATION_SYSTEMS = ['None', 'KOHA', 'SOUL', 'Other'] as const;

export const INFRASTRUCTURE_LABELS = {
  1: 'Very Poor',
  2: 'Poor',
  3: 'Average',
  4: 'Good',
  5: 'Excellent',
};

export const SATISFACTION_LABELS = {
  1: 'Very Dissatisfied',
  2: 'Dissatisfied',
  3: 'Neutral',
  4: 'Satisfied',
  5: 'Very Satisfied',
  6: 'Highly Satisfied',
  7: 'Extremely Satisfied',
  8: 'Outstanding',
  9: 'Exceptional',
  10: 'Perfect',
};

export const BARRIER_LABELS = {
  1: 'No Barrier',
  2: 'Minor Barrier',
  3: 'Moderate Barrier',
  4: 'Significant Barrier',
  5: 'Critical Barrier',
};

export const REPORT_TEMPLATES = [
  {
    id: 'executive-summary',
    name: 'Executive Summary',
    description: 'One-page overview for stakeholders',
    sections: ['summary', 'key-metrics', 'recommendations'],
  },
  {
    id: 'detailed-analysis',
    name: 'Detailed Analysis Report',
    description: 'Comprehensive 5-10 page analysis',
    sections: ['introduction', 'methodology', 'results', 'analysis', 'recommendations', 'conclusion'],
  },
  {
    id: 'thesis-chapter',
    name: 'Thesis Chapter',
    description: 'Academic format for thesis',
    sections: ['abstract', 'introduction', 'literature-review', 'methodology', 'results', 'discussion', 'conclusion', 'references'],
  },
  {
    id: 'funding-proposal',
    name: 'Funding Proposal',
    description: 'Proposal with budget tables',
    sections: ['executive-summary', 'needs-assessment', 'proposed-solution', 'budget', 'timeline', 'expected-outcomes'],
  },
  {
    id: 'policy-brief',
    name: 'Policy Brief',
    description: 'Government format policy document',
    sections: ['background', 'current-state', 'recommendations', 'implementation-plan', 'budget'],
  },
];

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const SUPABASE_CONFIG = {
  url: import.meta.env.VITE_SUPABASE_URL || '',
  key: import.meta.env.VITE_SUPABASE_KEY || '',
};
