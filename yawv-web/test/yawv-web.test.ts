import { html } from 'lit';
import { fixture, expect } from '@open-wc/testing';

import { YawvWeb } from '../src/YawvWeb.js';
import '../src/yawv-web.js';

describe('YawvWeb', () => {
  let element: YawvWeb;
  beforeEach(async () => {
    element = await fixture(html`<yawv-web></yawv-web>`);
  });

  it('renders a h1', () => {
    const h1 = element.shadowRoot!.querySelector('h1')!;
    expect(h1).to.exist;
    expect(h1.textContent).to.equal('My app');
  });

  it('passes the a11y audit', async () => {
    await expect(element).shadowDom.to.be.accessible();
  });
});
