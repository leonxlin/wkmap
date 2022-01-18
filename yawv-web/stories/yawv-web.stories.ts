import { html, TemplateResult } from 'lit';
import '../src/yawv-web.js';

export default {
  title: 'YawvWeb',
  component: 'yawv-web',
  argTypes: {
    backgroundColor: { control: 'color' },
  },
};

interface Story<T> {
  (args: T): TemplateResult;
  args?: Partial<T>;
  argTypes?: Record<string, unknown>;
}

interface ArgTypes {
  title?: string;
  backgroundColor?: string;
}

const Template: Story<ArgTypes> = ({
  title,
  backgroundColor = 'white',
}: ArgTypes) => html`
  <yawv-web
    style="--yawv-web-background-color: ${backgroundColor}"
    .title=${title}
  ></yawv-web>
`;

export const App = Template.bind({});
App.args = {
  title: 'My app',
};
