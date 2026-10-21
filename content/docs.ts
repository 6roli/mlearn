import materials from './materials.json';
import type { Origin } from '../src/content/types';

export const MATERIALS_DOCS = materials.documents as { id: string; lab: string; path: string; origin: Origin; title: string }[];
